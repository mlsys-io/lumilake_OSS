"""Standing contract-check suite for the dynamic workflow.

Every contract boundary in the dynamic workflow becomes a machine-checked
invariant here, so fix batches are verified automatically instead of costing a
review round. Each check is exhaustive by construction: it parametrizes over
the real constant (SUPPORTED_OPS, the runtime default map, the override
allowlist) rather than a hand-written list that can drift.

If any check fails on the current code, that is a real bug — do not weaken the
check to make it pass.
"""

import ast
import asyncio
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from lumid_hooks import PrincipalContext, ResourceRef

import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server import hooks
from lumilake_server.dynamic.driver import (
    _LIBRARY_REF_OVERRIDES,
    DriverProtocolError,
    _internal_id,
    compute_observation,
    observation_lambda,
    resolve_subgraph,
    system_message,
    validate_library,
)
from lumilake_server.middleware import TraceIdMiddleware
from lumilake_server.parser.yaml_parser import SUPPORTED_OPS, parse_yaml_payload
from lumilake_server.routes import jobs as job_routes_module
from lumilake_server.runtime.protocol import LumilakeResponse
from lumilake_server.utils.func_serialization import SAFE_BUILTINS, SAFE_MODULES
from lumilake_server.utils.job_storage import InMemoryJobStorage
from tests.server.test_dynamic.test_submission import (
    _SUBGRAPH_PLAN,
    _VALID_DYNAMIC_YAML,
    _run_background,
    _submit_body,
)

# Identical to _VALID_DYNAMIC_YAML but with a tiny job_timeout so a round that
# hangs past the deadline exercises the round-timeout cancellation path.
_TIMEOUT_DYNAMIC_YAML = """
name: dynamic
type: dynamic
goal: analyze market data
driver:
  model: Qwen/Qwen3-8B
  job_timeout: 0.05
  max_rounds: 4
  max_nodes_per_round: 4
  output_location:
    type: s3
    prefix: dynamic/data-free/
"""

# _VALID_DYNAMIC_YAML with NON-DEFAULT driver values (max_tokens, temperature,
# threshold) and a non-empty library block, so a dropped argument in either
# round-0 construction path produces a visibly different graph.
_ROUND0_DIFF_YAML = """
name: dynamic
type: dynamic
goal: analyze market data
driver:
  model: Qwen/Qwen3-32B
  max_tokens: 1234
  temperature: 0.9
  threshold: 0.7
  max_rounds: 4
  max_nodes_per_round: 4
  output_location:
    type: s3
    prefix: dynamic/data-free/
library:
  sector_market_cap:
    op: DataRetrievalOp
    data_spec:
      type: lumid
      mode: sql
      output_format: jsonl
      verify: false
      template: SELECT sector, AVG(market_cap) FROM reference.profile GROUP BY sector
"""

_DEMO_PRINCIPAL = PrincipalContext(
    principal_id="alice",
    org_id="demo",
    external_id="alice@example.com",
    principal_type="user",
    scopes=["admin"],
)


class _AllowAllIdentity:
    name = "test.identity"

    async def resolve(
        self, token: str, logger: logging.Logger
    ) -> PrincipalContext | None:
        return _DEMO_PRINCIPAL.model_copy(deep=True) if token == "token" else None


class _AllowAllPermissions:
    name = "test.permissions"

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return None

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        return None


class _AllowAllRegistrar:
    name = "test.registrar"

    async def register(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        return None

    async def deregister(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        return None


class _AllowAllGuards:
    name = "test.guards"

    async def check(self, principal: PrincipalContext, logger: logging.Logger) -> None:
        return None


class _FakeRuntimeManager:
    def set_dispatch_token(self, job_id: str, token: str | None) -> None:
        return None


class _FakeRuntimeServer:
    is_started = True

    def __init__(self) -> None:
        self.cancel_calls: list[str] = []
        self.parse_query_calls: list[dict[str, dict[str, Any]]] = []
        self.execute_calls: list[str] = []
        self.executed_graphs: list[dict[str, Any]] = []
        self.plans: list[dict[str, Any]] | None = None
        self.fail_rounds: set[int] = set()
        self.hang_rounds: set[int] = set()
        self.no_result_rounds: set[int] = set()
        self.hang_event = asyncio.Event()
        self.fail_cancel = False
        self.cancel_raises_cancelled = False
        self.omit_leaf_outputs = False
        self._traces: dict[str, list[str]] = {}
        self.runtime_manager = _FakeRuntimeManager()

    def parse_query(self, graph_specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.parse_query_calls.append(graph_specs)
        from lumilake_server.graphs import Graph

        return {
            name: Graph.from_json(spec["graph"]).compile(**spec["inputs"])
            for name, spec in graph_specs.items()
        }

    async def execute(
        self,
        graphs: dict[str, Any],
        request_id: str | None = None,
        config: Any | None = None,
        workflow_slices: dict[str, Any] | None = None,
    ) -> LumilakeResponse:
        self.execute_calls.append(request_id or "")
        self.executed_graphs.append(graphs)
        if request_id:
            self._traces[request_id] = [f"trace-{request_id}"]
        round_index = len(self.execute_calls) - 1
        if round_index in self.hang_rounds:
            await self.hang_event.wait()
            if self.cancel_raises_cancelled:
                from lumilake_server.runtime.protocol import RequestCancelledError

                raise RequestCancelledError(request_id or "")
        if round_index in self.fail_rounds:
            return LumilakeResponse(outputs={}, error_info=[{"message": "boom"}])
        if round_index in self.no_result_rounds:
            return LumilakeResponse(outputs={})
        plan = {"next": "STOP"}
        if self.plans:
            plan = self.plans[min(round_index, len(self.plans) - 1)]
        outputs: dict[str, Any] = {"round": {"plan": [json.dumps(plan)]}}
        if not self.omit_leaf_outputs:
            leaf_names = self._leaf_output_names(graphs)
            for name in leaf_names:
                outputs["round"][name] = ['{"market_cap": 10}']
        return LumilakeResponse(outputs=outputs)

    def _leaf_output_names(self, graphs: dict[str, Any]) -> list[str]:
        from lumilake_server.ops import OutputOp

        names: list[str] = []
        for graph in graphs.values():
            for op in graph.iter_ops(OutputOp):
                if str(op.name).startswith("leaf_"):
                    names.append(op.name)
        return names

    def trace_ids_for_request(self, job_id: str) -> list[str]:
        return list(self._traces.get(job_id, []))

    def optimization_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    def selection_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    def clustering_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    def release_request_workflows(self, job_id: str) -> None:
        self._traces.pop(job_id, None)

    async def cancel_request(self, job_id: str) -> None:
        self.cancel_calls.append(job_id)
        self.hang_event.set()
        if self.fail_cancel:
            raise RuntimeError("cancel RPC failed")

    async def preview_schedule(self, **kwargs: Any) -> Any:
        class _Schedule:
            worker_assignment: dict[str, Any] = {}

        class _Preview:
            request_id: str = "preview"
            selected_workers: list[str] = []
            schedule: Any = _Schedule()
            runtime_graph_node_counts: dict[str, int] = {}
            merged_runtime_node_count: int = 0
            selection_seconds: float = 0.0
            clustering_seconds: float = 0.0
            optimization_seconds: float = 0.0

        return _Preview()


@pytest.fixture(autouse=True)
def _reset_hook_state() -> Iterator[None]:
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.RESOURCE_REGISTRARS.clear()
    hooks.IDENTITY_PROVIDERS.append(_AllowAllIdentity())
    hooks.PERMISSION_CHECKERS.append(_AllowAllPermissions())
    hooks.SUBMISSION_GUARDS.append(_AllowAllGuards())
    hooks.RESOURCE_REGISTRARS.append(_AllowAllRegistrar())
    yield
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.RESOURCE_REGISTRARS.clear()
    hooks.USAGE_SINKS.clear()


@pytest.fixture
def job_routes(monkeypatch: pytest.MonkeyPatch) -> Any:
    storage = InMemoryJobStorage()
    job_storage_module._job_storage = storage
    job_routes_module.jobs.clear()
    job_routes_module._job_storage = storage
    fake_server = _FakeRuntimeServer()
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(lambda cls: fake_server),
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_instance",
        classmethod(lambda cls: fake_server),
    )
    monkeypatch.setattr(
        job_routes_module,
        "build_request_data_profile_tasks",
        lambda **kwargs: [],
    )

    async def _dump_output_locations(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        job_routes_module,
        "_dump_output_locations",
        _dump_output_locations,
    )
    setattr(job_routes_module, "_fake_runtime_server", fake_server)
    return job_routes_module


@pytest.fixture
def app(job_routes: Any) -> FastAPI:
    application = FastAPI()
    application.state.logger = logging.getLogger("test.dynamic_contracts")
    application.state.background_tasks = set()
    application.add_middleware(TraceIdMiddleware)
    application.include_router(job_routes.router)
    return application


_REPO_ROOT = Path(__file__).resolve().parents[3]
_OBSERVATION_TEMPLATE = (
    _REPO_ROOT / "src" / "lumilake_server" / "dynamic" / "observation_template.py"
)


# ---------------------------------------------------------------------------
# CHECK 1: sandbox name resolution
# ---------------------------------------------------------------------------


def _module_free_names(source: str) -> set[str]:
    """Return every name loaded (Load ctx) that is not locally bound.

    Locally bound names are module-level defs/assignments, function params,
    and names assigned anywhere in a function body.
    """
    tree = ast.parse(source)
    bound: set[str] = set()
    loaded: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.kwonlyargs:
                bound.add(arg.arg)
            if node.args.vararg:
                bound.add(node.args.vararg.arg)
            if node.args.kwarg:
                bound.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
    return loaded - bound


def test_sandbox_free_names_resolve_in_whitelist() -> None:
    """Every free name in the observation template must resolve in the sandbox.

    The template runs in the restricted sandbox; referencing any name outside
    SAFE_BUILTINS/SAFE_MODULES (e.g. ValueError, TypeError) raises NameError at
    runtime. This guards the P1 exception-handling class.
    """
    # Check the substituted source that actually runs in the sandbox (the
    # {width} placeholder is replaced before embedding).
    source = observation_lambda(preview_width=900)
    free = _module_free_names(source)
    whitelist = set(SAFE_BUILTINS) | set(SAFE_MODULES)
    unresolved = free - whitelist
    assert not unresolved, (
        f"observation template references names not in the sandbox whitelist: "
        f"{sorted(unresolved)}"
    )


def test_sandbox_observe_materializes_and_executes() -> None:
    """The generated observe() must materialize and run through the real
    sandbox call path (safe_materialize_function), not just parse."""
    from lumilake_server.utils.func_serialization import safe_materialize_function

    source = observation_lambda(preview_width=900)
    fn = safe_materialize_function(source)
    assert callable(fn)
    # A valid JSON record and a non-JSON string both run without raising.
    out = fn([['{"market_cap": 10}'], ["3 apples"]])
    assert isinstance(out, str)
    assert "rows=2" in out
    # Bare-string entries exercise the entry-level str branch (not a list).
    out = fn(["3 apples"])
    assert isinstance(out, str)
    assert "rows=1" in out
    out = fn(['{"m": 10}'])
    assert isinstance(out, str)
    assert "rows=1" in out


# ---------------------------------------------------------------------------
# CHECK 2: dataflow differential covers every op type
# ---------------------------------------------------------------------------
# The validator is derived from the parser (Root Cause A). For every op type
# that can carry a node reference, we assert BOTH directions (declared-but-
# unused rejected, native-but-undeclared rejected) plus a positive correctly-
# wired case. Op types with no config-based node reference (DataOp/LambdaOp/
# MessageOp) are exercised to confirm a declared input is accepted (closure-
# only wiring).


def _valid_op(op_type: str, ref: str) -> dict:
    """A parser-valid op of ``op_type`` wired to node ``ref``."""
    base: dict[str, Any] = {"id": "x", "op": op_type, "inputs": [ref]}
    if op_type == "DataRetrievalOp":
        base["data_spec"] = {
            "type": "lumid",
            "mode": "sql",
            "output_format": "jsonl",
            "verify": False,
            "template": "SELECT * FROM t WHERE id = {p}",
            "params": [{"label": "p", "node": ref}],
        }
    elif op_type in {"LLMChatOp", "LLMVisionOp"}:
        base["config"] = {"model": "Qwen/Qwen3-8B"}
        base["messages"] = [{"role": "user", "content": ref}]
        if op_type == "LLMVisionOp":
            base["image_source"] = ref
    elif op_type == "ImageGenerationOp":
        base["config"] = {"model": "Qwen/Qwen3-8B"}
        base["content"] = ref
    elif op_type == "EmbeddingOp":
        base["config"] = {"model": "Qwen/Qwen3-8B"}
        base["input"] = ref
    elif op_type == "FormatOp":
        base["template"] = "{ref0}"
        base["format_args"] = []
        base["format_kwargs"] = {"ref0": ref}
    elif op_type == "DataOp":
        base["data"] = ["x"]
    elif op_type == "MessageOp":
        base["messages"] = [{"role": "user", "content": ref}]
    elif op_type == "LambdaOp":
        base["fn_name"] = "fn"
        base["code"] = "def fn(args):\n    return 'x'\n"
    return base


# Op types whose native config wires a node dependency.
_REFERENCE_OP_TYPES = frozenset(
    {
        "DataRetrievalOp",
        "LLMChatOp",
        "LLMVisionOp",
        "ImageGenerationOp",
        "EmbeddingOp",
        "FormatOp",
    }
)

# Op types with no config-based node reference (inputs are closure-only).
_NO_REFERENCE_OP_TYPES = frozenset({"DataOp", "LambdaOp", "MessageOp"})


@pytest.mark.parametrize("op_type", sorted(SUPPORTED_OPS))
def test_every_op_type_is_classified(op_type: str) -> None:
    """Every SUPPORTED_OPS type is either a reference or no-reference type."""
    assert (
        op_type in _REFERENCE_OP_TYPES or op_type in _NO_REFERENCE_OP_TYPES
    ), f"op type {op_type!r} is in SUPPORTED_OPS but not classified"


def _stub(id: str) -> dict:
    return {"id": id, "op": "DataOp", "inputs": [], "data": ["x"]}


def _consumer() -> dict:
    """An LLMChatOp that consumes the tested op so it is not a leaf."""
    return {
        "id": "consumer",
        "op": "LLMChatOp",
        "inputs": ["x"],
        "config": {"model": "Qwen/Qwen3-8B"},
        "messages": [{"role": "user", "content": "x"}],
    }


@pytest.mark.parametrize("op_type", sorted(_REFERENCE_OP_TYPES))
def test_declared_but_unused_input_rejected(op_type: str) -> None:
    """A declared input the native config does not consume is rejected."""
    from lumilake_server.dynamic.driver import validate_emitted_subgraph

    op = _valid_op(op_type, "a")
    op["inputs"] = ["a", "b"]  # declare b but config only references a
    with pytest.raises(Exception, match="does not reference"):
        validate_emitted_subgraph([_stub("a"), _stub("b"), op, _consumer()], {}, 8)


# Op types whose config can carry more than one node reference, so an
# undeclared ref can be pulled into the probe transitively and only the
# validator's native-but-undeclared loop catches it. Single-ref types
# (ImageGenerationOp/EmbeddingOp) always die at the parser instead.
_MULTI_REF_OP_TYPES = frozenset(
    {"DataRetrievalOp", "LLMChatOp", "LLMVisionOp", "FormatOp"}
)


@pytest.mark.parametrize("op_type", sorted(_MULTI_REF_OP_TYPES))
def test_native_but_undeclared_reference_rejected(op_type: str) -> None:
    """A native reference missing from declared inputs is rejected.

    The undeclared ref is pulled into the probe transitively (a declared op
    references it), so the parser resolves it and only the validator's
    native-but-undeclared loop can catch it.
    """
    from lumilake_server.dynamic.driver import validate_emitted_subgraph

    op = _valid_op(op_type, "a")
    op["inputs"] = ["a"]  # declare a, but config also references b
    op = _add_second_ref(op, "b")
    # b is pulled into the probe transitively because a consumes b.
    a = _valid_op(op_type, "b")
    a["id"] = "a"
    a["inputs"] = ["b"]
    with pytest.raises(Exception, match="missing from the declared inputs"):
        validate_emitted_subgraph([_stub("b"), a, op, _consumer()], {}, 8)


def _add_second_ref(op: dict, ref: str) -> dict:
    """Add a second native config reference to a reference op."""
    op_type = op["op"]
    if op_type == "DataRetrievalOp":
        op["data_spec"]["params"].append({"label": "p2", "node": ref})
    elif op_type in {"LLMChatOp", "LLMVisionOp"}:
        op["messages"].append({"role": "user", "content": ref})
    elif op_type == "ImageGenerationOp":
        op["content"] = f"{op['content']} {ref}"
    elif op_type == "EmbeddingOp":
        op["input"] = f"{op['input']} {ref}"
    elif op_type == "FormatOp":
        op["format_kwargs"]["ref1"] = ref
    return op


@pytest.mark.parametrize("op_type", sorted(_REFERENCE_OP_TYPES))
def test_correctly_wired_op_accepted(op_type: str) -> None:
    """A correctly-wired op (declared == native) is accepted."""
    from lumilake_server.dynamic.driver import validate_emitted_subgraph

    op = _valid_op(op_type, "a")
    validate_emitted_subgraph([_stub("a"), op, _consumer()], {}, 8)


@pytest.mark.parametrize("op_type", sorted(_NO_REFERENCE_OP_TYPES))
def test_no_reference_op_accepts_declared_input(op_type: str) -> None:
    """Op types with no config-based node reference accept a declared input."""
    from lumilake_server.dynamic.driver import validate_emitted_subgraph

    op = _valid_op(op_type, "a")
    validate_emitted_subgraph([_stub("a"), op, _consumer()], {}, 8)


def _message_op(id: str) -> dict:
    return {
        "id": id,
        "op": "MessageOp",
        "inputs": [],
        "messages": [{"role": "user", "content": "hi"}],
    }


def _llm_messages_ref(id: str, ref: str) -> dict:
    return {
        "id": id,
        "op": "LLMChatOp",
        "inputs": [ref],
        "messages_ref": ref,
        "config": {"model": "Qwen/Qwen3-8B"},
    }


def _consumer_of(id: str) -> dict:
    """An LLMChatOp that consumes ``id`` so it is not a leaf."""
    return {
        "id": "consumer",
        "op": "LLMChatOp",
        "inputs": [id],
        "config": {"model": "Qwen/Qwen3-8B"},
        "messages": [{"role": "user", "content": id}],
    }


def test_same_round_messages_ref_accepted() -> None:
    """A same-round MessageOp referenced by messages_ref is accepted."""
    from lumilake_server.dynamic.driver import validate_emitted_subgraph

    validate_emitted_subgraph(
        [_message_op("m"), _llm_messages_ref("L", "m"), _consumer_of("L")], {}, 8
    )


def test_cross_round_messages_ref_accepted() -> None:
    """A prior-round MessageOp referenced by messages_ref is accepted.

    The prior-round op is in the registry with its real type, so the
    type-constrained reference resolves instead of falling back to a stub.
    """
    from lumilake_server.dynamic.driver import validate_emitted_subgraph

    registry = {"prior_message": _message_op("prior_message")}
    validate_emitted_subgraph(
        [_llm_messages_ref("L", "prior_message"), _consumer_of("L")], registry, 8
    )


def _retrieval_ref(id: str, ref: str) -> dict:
    return {
        "id": id,
        "op": "DataRetrievalOp",
        "inputs": [ref],
        "data_spec": {
            "type": "lumid",
            "mode": "sql",
            "output_format": "jsonl",
            "verify": False,
            "template": "SELECT * FROM t WHERE id = {p}",
            "params": [{"label": "p", "node": ref}],
        },
    }


def test_cross_round_native_but_undeclared_rejected() -> None:
    """A prior-round op referenced by config but missing from declared inputs
    is rejected, mirroring the same-round direction."""
    from lumilake_server.dynamic.driver import validate_emitted_subgraph

    # pb is pulled into the probe transitively because pa consumes pb, so the
    # parser resolves it and only the validator loop catches c's undeclared ref.
    registry = {
        "pa": _retrieval_ref("pa", "pb"),
        "pb": _retrieval_ref("pb", "Symbols"),
    }
    # c declares only pa, but its config references pa AND pb.
    c = _retrieval_ref("c", "pa")
    c["data_spec"]["params"].append({"label": "p2", "node": "pb"})
    with pytest.raises(Exception, match="missing from the declared inputs"):
        validate_emitted_subgraph([c, _consumer_of("c")], registry, 8)


def test_cross_round_declared_but_unused_rejected() -> None:
    """A declared prior-round input the config never consumes is rejected,
    mirroring the same-round direction."""
    from lumilake_server.dynamic.driver import validate_emitted_subgraph

    registry = {
        "pa": _retrieval_ref("pa", "Symbols"),
        "pb": _retrieval_ref("pb", "Symbols"),
    }
    # c declares pa and pb, but its config references only pa.
    c = _retrieval_ref("c", "pa")
    c["inputs"] = ["pa", "pb"]
    with pytest.raises(Exception, match="does not reference"):
        validate_emitted_subgraph([c, _consumer_of("c")], registry, 8)


# ---------------------------------------------------------------------------
# CHECK 3: leaf archive path matches the runtime default
# ---------------------------------------------------------------------------


def _runtime_default_path_map() -> dict[str, str]:
    """Read the runtime's inline default-path map so a change breaks loudly."""
    import re

    from lumilake_server.runtime import runtime_graph

    source = Path(runtime_graph.__file__).read_text()
    match = re.search(r"default_path\s*=\s*\{([^}]*)\}", source, re.DOTALL)
    assert match, "runtime default_path map not found in runtime_graph.py"
    body = match.group(1)
    entries = re.findall(r'"(\w+)":\s*"([^"]+)"', body)
    return dict(entries)


def test_runtime_default_path_map_is_pinned() -> None:
    """Pin the runtime's per-mode default output paths."""
    assert _runtime_default_path_map() == {
        "sql": "items.table",
        "s3": "items.content",
        "agent": "items.table",
    }


@pytest.mark.parametrize("mode", ["sql", "s3", "agent"])
def test_retrieval_leaf_outputop_has_no_path_override(mode: str) -> None:
    """A retrieval leaf OutputOp must carry no path so the runtime applies its
    mode-derived default."""
    from lumilake_server.dynamic.driver import build_round

    subgraph = [
        {
            "id": "q1",
            "op": "DataRetrievalOp",
            "inputs": ["Symbols"],
            "data_spec": {
                "type": "lumid",
                "mode": mode,
                "output_format": "jsonl",
                "verify": False,
                "template": "SELECT * FROM x",
                "params": [{"label": "s", "node": "Symbols"}],
            },
        }
    ]
    round_build = build_round(
        subgraph,
        node_registry={},
        round_index=1,
        goal="g",
        observations=[],
        topology=[],
        preview_width=900,
        model="Qwen/Qwen3-8B",
        max_tokens=768,
        temperature=0.4,
    )
    graph = round_build.graph
    leaf_outputs = [
        v
        for v in graph.values()
        if v.get("_op") == "OutputOp" and str(v.get("name", "")).startswith("leaf_")
    ]
    assert leaf_outputs, "expected at least one leaf OutputOp"
    for output in leaf_outputs:
        assert "path" not in output, (
            f"leaf OutputOp must not set a path override (mode={mode}); "
            "the runtime applies its own default"
        )


def test_llm_leaf_outputop_has_no_path_override() -> None:
    """An LLM leaf OutputOp must carry no path so the runtime defaults to
    items.output."""
    from lumilake_server.dynamic.driver import build_round

    subgraph: list[dict[str, Any]] = [
        {
            "id": "a",
            "op": "DataRetrievalOp",
            "inputs": ["Symbols"],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "output_format": "jsonl",
                "verify": False,
                "template": "SELECT * FROM x",
                "params": [{"label": "s", "node": "Symbols"}],
            },
        },
        {
            "id": "b",
            "op": "LLMChatOp",
            "inputs": ["a"],
            "config": {"model": "Qwen/Qwen3-8B"},
            "messages": [{"role": "user", "content": "a"}],
        },
    ]
    round_build = build_round(
        subgraph,
        node_registry={},
        round_index=1,
        goal="g",
        observations=[],
        topology=[],
        preview_width=900,
        model="Qwen/Qwen3-8B",
        max_tokens=768,
        temperature=0.4,
    )
    graph = round_build.graph
    leaf_outputs = [
        v
        for v in graph.values()
        if v.get("_op") == "OutputOp" and str(v.get("name", "")).startswith("leaf_")
    ]
    assert leaf_outputs
    for output in leaf_outputs:
        assert "path" not in output


# NOTE (Check 3 known uncovered boundary): these checks assert the leaf
# OutputOp carries no path override, pin the runtime's default-path map, and
# guard the ADAPTER boundary (_extract_leaf_outputs rejects any element that is
# not the documented single-string archive representation). They do NOT prove
# the real archive backend returns that representation — only a live-backend
# round trip could, and that remains uncovered.


def test_extract_leaf_outputs_rejects_decoded_element() -> None:
    """A decoded (non-string) archive element must be rejected, not str()-ed."""
    from lumilake_server.dynamic.driver import DriverProtocolError
    from lumilake_server.routes.jobs import _extract_leaf_outputs

    # A decoded dict element would str() to a repr and silently degrade.
    with pytest.raises(DriverProtocolError, match="must be a list containing"):
        _extract_leaf_outputs({"round": {"leaf_a": [{"market_cap": 10}]}}, ["leaf_a"])
    # A decoded list element is likewise rejected.
    with pytest.raises(DriverProtocolError, match="must be a list containing"):
        _extract_leaf_outputs({"round": {"leaf_a": [[1, 2]]}}, ["leaf_a"])
    # The documented single-string representation still passes.
    out = _extract_leaf_outputs(
        {"round": {"leaf_a": ['{"market_cap": 10}']}}, ["leaf_a"]
    )
    assert out == {"a": ['{"market_cap": 10}']}


# ---------------------------------------------------------------------------
# CHECK 4: observation differential (in-graph vs server recompute)
#   4a test_observation_shape_differential: compute_observation agrees with the
#      generated observe() on a given nested shape (shape invariant).
#   4b test_observation_order_differential: build_round feeds the observation
#      LambdaOp leaves in the same sorted order compute_observation consumes
#      (order invariant, guards the P1-4 defect).
# ---------------------------------------------------------------------------


def _generated_observe(preview_width: int):
    """Materialize the exact observe() the in-graph LambdaOp receives."""
    source = observation_lambda(preview_width=preview_width)
    namespace: dict = {"json": json}
    exec(source, namespace)  # noqa: S102 - sandbox-safe aggregation template
    return namespace["observe"]


@pytest.mark.parametrize(
    "leaf_outputs",
    [
        # Single JSON record.
        {"leaf1": ['{"market_cap": 10}']},
        # Many JSON records.
        {"leaf1": ['{"market_cap": 10}', '{"market_cap": 20}']},
        # A list of JSON strings.
        {"leaf1": ['["a", "b"]']},
        # Plain LLM text.
        {"leaf1": ["hello world"]},
        # Malformed JSON treated as text.
        {"leaf1": ["not json {"]},
        # Non-JSON strings that START like JSON (digit/quote/bracket/brace) —
        # these pass a shape guard but must be caught by json.JSONDecodeError
        # and treated as text, not raise.
        {"leaf1": ["3 apples"]},
        {"leaf1": ["-1 degrees celsius"]},
        {"leaf1": ['"unterminated']},
        {"leaf1": ["[incomplete"]},
        {"leaf1": ["{not: valid}"]},
        # Two leaves whose ids sort differently from emitted order.
        {"z_result": ['{"x": 5}'], "a_result": ['{"x": 1}']},
    ],
)
def test_observation_shape_differential(leaf_outputs: dict) -> None:
    """compute_observation must equal calling the generated observe() directly.

    Guards the SHAPE invariant: the server recompute and the generated in-graph
    observe() must agree on the same nested argument shape and ordering for a
    given leaf set. This does NOT exercise build_round; ordering drift between
    the in-graph path and compute_observation is guarded by
    test_observation_order_differential.
    """
    width = 900
    server_obs = compute_observation(leaf_outputs, preview_width=width)
    observe = _generated_observe(width)
    per_leaf = [leaf_outputs[k] for k in sorted(leaf_outputs)]
    direct_obs = observe(per_leaf)
    assert server_obs == direct_obs


def test_observation_order_differential() -> None:
    """The in-graph observation leaf order must match compute_observation's.

    Guards the ORDER invariant (the P1-4 defect): build_round must feed the
    observation LambdaOp leaves in the same deterministic (sorted) order that
    compute_observation consumes. If ``sorted()`` is removed from leaf_ids in
    build_round, the in-graph order diverges from compute_observation's sorted
    order and this check fails.
    """
    from lumilake_server.dynamic.driver import build_round

    def _retrieval(id: str) -> dict:
        return {
            "id": id,
            "op": "DataRetrievalOp",
            "inputs": ["Symbols"],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "output_format": "jsonl",
                "verify": False,
                "template": "SELECT * FROM x",
                "params": [{"label": "s", "node": "Symbols"}],
            },
        }

    # Emit leaves in an order that differs from sorted order.
    subgraph = [_retrieval("z_result"), _retrieval("a_result")]
    round_build = build_round(
        subgraph,
        node_registry={},
        round_index=1,
        goal="g",
        observations=[],
        topology=[],
        preview_width=900,
        model="Qwen/Qwen3-8B",
        max_tokens=768,
        temperature=0.4,
    )
    graph = round_build.graph

    # Extract the observation LambdaOp's _inputs (the in-graph leaf order, as
    # internal ids).
    observation = graph.get("observation")
    assert observation is not None, "round graph has no observation node"
    assert observation.get("_op") == "LambdaOp"
    # Each leaf is observed through its render_<leaf> FormatOp, so strip that
    # prefix to recover the leaf order the observation actually sees.
    in_graph_internal = [
        node_id[len("render_") :] for node_id in observation["_inputs"]
    ]

    # round_build.leaf_output_names are leaf_<internal id> in the same order
    # build_round produced; strip the prefix to get internal ids.
    built_internal = [name[len("leaf_") :] for name in round_build.leaf_output_names]
    assert (
        in_graph_internal == built_internal
    ), "observation _inputs do not match the round's leaf output names"

    # compute_observation consumes sorted(leaf_outputs); the in-graph order must
    # be the same sorted order.
    assert in_graph_internal == sorted(in_graph_internal), (
        "in-graph observation leaf order is not sorted; it diverges from "
        "compute_observation's sorted order"
    )
    assert round_build.leaf_output_names == sorted(round_build.leaf_output_names)


# ---------------------------------------------------------------------------
# CHECK 7: heuristic drift detection over planner-facing prose
# ---------------------------------------------------------------------------
# This is NOT a proof that the docs express the allowlist contract. It is
# heuristic drift detection: it scans planner-facing prose for claim verbs
# followed by a field word and asserts any such field is in
# _LIBRARY_REF_OVERRIDES. It cannot catch every phrasing (e.g. "supports
# setting custom params", or deleting the restriction paragraph outright), so
# it is a tripwire, not a guarantee. Widening the allowlist updates the check
# automatically; an unlisted field that uses a recognized verb is caught.
# The claim verbs all express the same promise (override / set / change /
# supply / provide / specify / customise / edit / modify / replace / adjust /
# configure); each captures the field word. Markdown formatting characters
# around the field word (backticks, asterisks, underscores) are stripped
# before matching.
_OVERRIDE_CLAIM_RE = re.compile(
    r"\b(?:overrid(?:e|ing)|set|change|supply|provide|specify|customi[sz]e|"
    r"edit|modify|replace|adjust|configure)\s+(?:the\s+)?([*_`]*\w+[*_`]*)\b",
    re.IGNORECASE,
)


def _assert_no_forbidden_override(text: str, source: str) -> None:
    for word in _OVERRIDE_CLAIM_RE.findall(text):
        field = word.strip("*_`").lower()
        assert field in _LIBRARY_REF_OVERRIDES, (
            f"{source} promises override of {field!r} outside the allowlist "
            f"{sorted(_LIBRARY_REF_OVERRIDES)}"
        )


def test_system_message_does_not_promise_overrides_beyond_allowlist() -> None:
    """The planner-facing prompt must not promise overrides outside the
    enforced allowlist."""
    library = {
        "top_peers": {
            "op": "DataRetrievalOp",
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "output_format": "jsonl",
                "verify": False,
                "template": "SELECT * FROM reference.peers WHERE symbol = {symbol}",
                "params": [{"label": "symbol", "node": "Symbols"}],
                "param_bindings": [{"label": "symbol", "input": 0}],
            },
        }
    }
    msg = system_message("goal", [], [], library=library)
    _assert_no_forbidden_override(msg, "system_message")


def test_example_yaml_does_not_promise_overrides_beyond_allowlist() -> None:
    """The example YAML comment must not claim overridable fields outside the
    enforced allowlist."""
    example = (
        _REPO_ROOT / "examples" / "templates" / "yaml" / "market-data-dynamic.yaml"
    )
    text = example.read_text()
    _assert_no_forbidden_override(text, "example YAML")


def test_example_yaml_loads_and_library_validates() -> None:
    """The shipped example YAML must load through the dynamic spec loader and
    its library must pass validate_library."""
    from lumilake_server.dynamic.spec import load_spec

    example = (
        _REPO_ROOT / "examples" / "templates" / "yaml" / "market-data-dynamic.yaml"
    )
    spec = load_spec(example)
    assert spec.library is not None
    validate_library(spec.library)
    # The cross-round entries the example advertises are present.
    assert "top_sector" in spec.library
    assert "peers_in_sector" in spec.library


def test_sdk_docs_do_not_promise_overrides_beyond_allowlist() -> None:
    """The SDK docs' dynamic-workflow section must not claim overridable fields
    outside the enforced allowlist."""
    sdk = (_REPO_ROOT / "docs" / "SDK.md").read_text()
    # Scope to the dynamic-workflow section: from "### Dynamic workflows" to
    # the next "## " header, so unrelated prose (e.g. the client-timeout
    # "Override it three ways") does not false-positive.
    start = sdk.index("### Dynamic workflows")
    end = sdk.index("\n## ", start)
    _assert_no_forbidden_override(sdk[start:end], "docs/SDK.md dynamic section")


# ---------------------------------------------------------------------------
# CHECK 5: submit/preview differential
# ---------------------------------------------------------------------------

_INVALID_DYNAMIC_ENVELOPES = [
    # Multi-entry data list.
    {
        "data": [
            {
                "name": "a",
                "workflow": _VALID_DYNAMIC_YAML,
                "inputs": {"Symbols": ["NVDA"]},
            },
            {
                "name": "b",
                "workflow": _VALID_DYNAMIC_YAML,
                "inputs": {"Symbols": ["NVDA"]},
            },
        ]
    },
    # Empty-string symbol.
    {
        "data": [
            {
                "name": "demo",
                "workflow": _VALID_DYNAMIC_YAML,
                "inputs": {"Symbols": [""]},
            }
        ]
    },
    # Whitespace-only symbol.
    {
        "data": [
            {
                "name": "demo",
                "workflow": _VALID_DYNAMIC_YAML,
                "inputs": {"Symbols": ["   "]},
            }
        ]
    },
    # Multi-symbol.
    {
        "data": [
            {
                "name": "demo",
                "workflow": _VALID_DYNAMIC_YAML,
                "inputs": {"Symbols": ["NVDA", "AAPL"]},
            }
        ]
    },
    # Driver output_location of type db.
    {
        "data": [
            {
                "name": "demo",
                "workflow": _VALID_DYNAMIC_YAML.replace(
                    "    type: s3\n    prefix: dynamic/data-free/",
                    "    type: db\n    table: schema.tbl\n    column: col",
                ),
                "inputs": {"Symbols": ["NVDA"]},
            }
        ]
    },
    # Per-entry output_location of type db (no driver location, so the
    # per-entry one is effective).
    {
        "data": [
            {
                "name": "demo",
                "workflow": _VALID_DYNAMIC_YAML.replace(
                    "  output_location:\n"
                    "    type: s3\n"
                    "    prefix: dynamic/data-free/\n",
                    "",
                ),
                "inputs": {"Symbols": ["NVDA"]},
                "output_location": {
                    "type": "db",
                    "table": "schema.tbl",
                    "column": "col",
                },
            }
        ]
    },
    # Non-default poll_interval (rejected by the shared renderer).
    {
        "data": [
            {
                "name": "demo",
                "workflow": _VALID_DYNAMIC_YAML.replace(
                    "  model: Qwen/Qwen3-8B\n",
                    "  model: Qwen/Qwen3-8B\n  poll_interval: 5.0\n",
                ),
                "inputs": {"Symbols": ["NVDA"]},
                "output_location": {"type": "s3", "prefix": "dynamic/data-free/"},
            }
        ]
    },
    # Library entry with an unsupported op type (rejected by the shared
    # renderer).
    {
        "data": [
            {
                "name": "demo",
                "workflow": _VALID_DYNAMIC_YAML + "library:\n  bad:\n    op: NotAnOp\n",
                "inputs": {"Symbols": ["NVDA"]},
                "output_location": {"type": "s3", "prefix": "dynamic/data-free/"},
            }
        ]
    },
]


@pytest.mark.anyio
@pytest.mark.parametrize("envelope", _INVALID_DYNAMIC_ENVELOPES)
async def test_submit_and_preview_reject_same_invalid_dynamic(
    app: Any, envelope: dict
) -> None:
    """Both doors (submit and preview) must reject the identical invalid set."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        submit_resp = await client.post(
            "/jobs",
            json=envelope,
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
        preview_resp = await client.post(
            "/jobs/preview",
            json=envelope,
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert (
        submit_resp.status_code == 422
    ), f"submit accepted invalid dynamic envelope: {submit_resp.text}"
    assert (
        preview_resp.status_code == 422
    ), f"preview accepted invalid dynamic envelope: {preview_resp.text}"


_VALID_DYNAMIC_ENVELOPES = [
    # Standard valid envelope.
    _submit_body(_VALID_DYNAMIC_YAML),
    # S3 driver location plus a shadowed entry-level DB location: the driver
    # location takes precedence, so the DB is ignored and both doors accept.
    {
        "data": [
            {
                "name": "demo",
                "workflow": _VALID_DYNAMIC_YAML,
                "inputs": {"Symbols": ["NVDA"]},
                "output_location": {
                    "type": "db",
                    "table": "schema.tbl",
                    "column": "col",
                },
            }
        ]
    },
]


@pytest.mark.anyio
@pytest.mark.parametrize("envelope", _VALID_DYNAMIC_ENVELOPES)
async def test_submit_and_preview_accept_same_valid_dynamic(
    app: Any, envelope: dict
) -> None:
    """Both doors (submit and preview) must accept the identical valid set."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        submit_resp = await client.post(
            "/jobs",
            json=envelope,
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
        preview_resp = await client.post(
            "/jobs/preview",
            json=envelope,
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert (
        submit_resp.status_code == 200
    ), f"submit rejected valid dynamic envelope: {submit_resp.text}"
    assert (
        preview_resp.status_code == 200
    ), f"preview rejected valid dynamic envelope: {preview_resp.text}"


@pytest.mark.anyio
async def test_submit_and_preview_derive_same_effective_location(
    app: Any, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both doors must resolve the SAME effective output location for the same
    envelope, not merely return the same status. Preview does not expose its
    derived location, so spy on the shared helper to capture what each door
    resolves."""
    import lumilake_server.routes.jobs as jobs_module

    resolved: list[dict] = []
    real_helper = jobs_module._effective_dynamic_output_location

    def _spy(dynamic_loc: Any, entry_loc: Any) -> Any:
        result = real_helper(dynamic_loc, entry_loc)
        resolved.append(result.model_dump())
        return result

    monkeypatch.setattr(jobs_module, "_effective_dynamic_output_location", _spy)
    # Driver and entry locations DIFFER, so a precedence break on one door
    # yields a different resolved value and the spy catches it.
    envelope = _submit_body(_VALID_DYNAMIC_YAML)
    envelope["data"][0]["output_location"] = {
        "type": "s3",
        "prefix": "dynamic/entry-prefix/",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        submit_resp = await client.post(
            "/jobs",
            json=envelope,
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
        preview_resp = await client.post(
            "/jobs/preview",
            json=envelope,
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert submit_resp.status_code == 200, submit_resp.text
    assert preview_resp.status_code == 200, preview_resp.text
    # Both doors must have resolved the effective location identically.
    assert len(resolved) == 2, f"expected 2 resolutions, got {len(resolved)}"
    assert resolved[0] == resolved[1], (
        f"submit and preview derived different effective locations: "
        f"{resolved[0]} vs {resolved[1]}"
    )
    # The location actually stored on the submitted parent must match what the
    # helper resolved — the helper result must flow through to the record.
    job_id = submit_resp.json()["data"]["job_id"]
    parent = job_routes.jobs[job_id]
    stored = next(iter(parent.output_location.values())).model_dump()
    assert stored == resolved[0], (
        f"stored parent location {stored} does not match helper-resolved "
        f"{resolved[0]}"
    )


# ---------------------------------------------------------------------------
# CHECK 6: loop terminal-state coverage
# ---------------------------------------------------------------------------
# NOTE (Check 6 known uncovered boundary): this family runs against
# InMemoryJobStorage and asserts the in-memory parent record's terminal status.
# It does NOT assert that the terminal status is durably persisted (a durable
# record could stay stale while the in-memory parent is terminal), nor that the
# parent's terminal hooks fire exactly once. The test below spies on the
# storage save calls and the usage sinks to cover both cheaply and hermetically.


@pytest.mark.anyio
async def test_loop_build_round_failure_reaches_terminal(
    app: Any, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_round raising must leave the parent failed, not running."""
    import lumilake_server.routes.jobs as jobs_module

    real_build_round = jobs_module.build_round

    def _boom(*args: Any, **kwargs: Any) -> Any:
        # Round 0 is built at submission time; only fail the loop's rounds.
        if kwargs.get("round_index", 0) >= 1:
            raise ValueError("build failed")
        return real_build_round(*args, **kwargs)

    monkeypatch.setattr(jobs_module, "build_round", _boom)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    # Round 0 returns a subgraph so the loop reaches round 1, whose build_round
    # raises.
    fake_server.plans = [_SUBGRAPH_PLAN]
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert (
        record.status == "failed"
    ), f"build_round failure must leave the parent failed, got {record.status!r}"


@pytest.mark.anyio
async def test_loop_child_failure_reaches_terminal(app: Any, job_routes: Any) -> None:
    """A failed child round must leave the parent failed."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    fake_server.fail_rounds = {1}
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert (
        record.status == "failed"
    ), f"child failure must leave the parent failed, got {record.status!r}"


@pytest.mark.anyio
async def test_loop_child_cancelled_reaches_terminal(
    app: Any, job_routes: Any, wait_for_inflight_child: Any
) -> None:
    """A child cancelled mid-loop must leave the parent cancelled."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    # Genuinely cancel a CHILD mid-loop: hang round 1's child, cancel the child
    # (not the parent), then release the hang so the child's _run_job returns.
    fake_server.hang_rounds = {1}
    fake_server.cancel_raises_cancelled = True
    loop_task = asyncio.create_task(_run_background(app))
    # Wait for the round-1 child (the one that hangs) to be genuinely in flight.
    child_id = await wait_for_inflight_child(job_routes, job_id)
    assert job_routes.jobs[job_id].status == "running"
    job_routes.jobs[child_id].status = "cancelled"
    fake_server.hang_event.set()
    await loop_task
    record = job_routes.jobs[job_id]
    assert (
        record.status == "cancelled"
    ), f"child cancellation must leave the parent cancelled, got {record.status!r}"


@pytest.mark.anyio
async def test_loop_child_no_result_reaches_terminal(app: Any, job_routes: Any) -> None:
    """A child returning no result must leave the parent failed."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    fake_server.no_result_rounds = {1}
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "failed", f"expected parent failed, got {record.status!r}"


@pytest.mark.anyio
async def test_loop_invalid_plan_reaches_terminal(app: Any, job_routes: Any) -> None:
    """An invalid plan must leave the parent failed."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [{"next": "bogus"}]
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "failed", f"expected parent failed, got {record.status!r}"


@pytest.mark.anyio
async def test_loop_missing_leaf_reaches_terminal(app: Any, job_routes: Any) -> None:
    """A missing expected leaf output must leave the parent failed."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    plan = json.loads(json.dumps(_SUBGRAPH_PLAN))
    plan["ops"][0]["id"] = "q2"
    fake_server.plans = [plan, {"next": "STOP"}]
    fake_server.omit_leaf_outputs = True
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "failed", f"expected parent failed, got {record.status!r}"


@pytest.mark.anyio
async def test_loop_register_resource_failure_reaches_terminal(
    app: Any, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """register_resource raising during child setup must leave the parent
    failed, not running."""
    import lumilake_server.routes.jobs as jobs_module

    real_register = jobs_module.register_resource
    calls = {"n": 0}

    async def _boom(*args: Any, **kwargs: Any) -> None:
        # The parent is registered at submission time; fail the child-setup
        # registration during the loop.
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("register failed")
        await real_register(*args, **kwargs)

    monkeypatch.setattr(jobs_module, "register_resource", _boom)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "failed", f"expected parent failed, got {record.status!r}"


@pytest.mark.anyio
async def test_loop_storage_save_failure_reaches_terminal(
    app: Any, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job-storage save raising must leave the parent failed, not running."""
    import lumilake_server.routes.jobs as jobs_module

    real_save = jobs_module._job_storage.save
    calls = {"n": 0}

    def _boom(*args: Any, **kwargs: Any) -> None:
        # The parent is saved at submission time; fail a save during the loop.
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("save failed")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(jobs_module._job_storage, "save", _boom)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    # A save failure can escape the background task; the in-memory parent must
    # still be terminal.
    try:
        await _run_background(app)
    except Exception:  # noqa: BLE001 - the save failure may propagate
        pass
    record = job_routes.jobs[job_id]
    assert record.status == "failed", f"expected parent failed, got {record.status!r}"


@pytest.mark.anyio
async def test_loop_terminal_status_persisted_and_hooks_fire_once(
    app: Any, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent's terminal status is durably saved and terminal hooks fire
    exactly once."""
    import lumilake_server.routes.jobs as jobs_module
    from lumilake_server.hooks import USAGE_SINKS

    saved_statuses: list[str] = []
    real_save = jobs_module._job_storage.save
    target: dict[str, str] = {}

    def _spy_save(record: Any) -> None:
        if target and record.job_id == target["id"]:
            saved_statuses.append(record.status)
        real_save(record)

    monkeypatch.setattr(jobs_module._job_storage, "save", _spy_save)

    class _UsageSpy:
        name = "test.usage_spy"
        calls = 0

        async def emit(self, rows: Any, logger: Any) -> None:
            _UsageSpy.calls += 1

    USAGE_SINKS.append(_UsageSpy())

    # Spy on trace-resource registration: _fire_parent_terminal_hooks registers
    # each trace id as a TRACE resource exactly once per terminal parent.
    from lumilake_server.hooks import ResourceKind

    trace_regs: list[str] = []
    trace_metas: list[dict] = []
    real_register = jobs_module.register_resource

    async def _spy_register(
        principal: Any, kind: Any, rid: Any, meta: Any, logger: Any
    ) -> None:
        if kind == ResourceKind.TRACE:
            trace_regs.append(rid)
            trace_metas.append(dict(meta))
        await real_register(principal, kind, rid, meta, logger)

    monkeypatch.setattr(jobs_module, "register_resource", _spy_register)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    target["id"] = job_id
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "completed"
    # The parent must have been saved at least once in its terminal state.
    # This guards the CONTRACT (terminal status is durably persisted), not a
    # specific save site: the parent is saved on two paths (the loop's own save
    # and the terminal-hook save), so a single-path mutation surviving is
    # expected — only skipping both breaks it.
    assert (
        "completed" in saved_statuses
    ), f"parent terminal status never durably saved; saw {saved_statuses!r}"
    # Terminal hooks (usage emission) fire exactly once.
    assert (
        _UsageSpy.calls == 1
    ), f"terminal hooks fired {_UsageSpy.calls} times, expected exactly once"
    # Trace resources are registered exactly once per child trace id (one per
    # child), with no duplicates.
    assert len(trace_regs) == len(record.child_job_ids), (
        f"trace resources registered {len(trace_regs)} times, expected "
        f"{len(record.child_job_ids)} (one per child)"
    )
    assert len(set(trace_regs)) == len(
        trace_regs
    ), f"trace resources registered with duplicates: {trace_regs}"
    # The registered trace ids must equal the parent's aggregated trace_ids,
    # and each registration's metadata must identify the correct parent.
    assert set(trace_regs) == set(
        record.trace_ids
    ), f"registered trace ids {trace_regs} != parent aggregated {record.trace_ids}"
    assert all(
        meta.get("job_id") == job_id for meta in trace_metas
    ), f"trace registration metadata does not name the parent: {trace_metas}"


@pytest.mark.anyio
async def test_loop_failed_terminal_persisted_and_hooks_fire_once(
    app: Any, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FAILED parent's terminal status is durably saved and terminal hooks
    fire exactly once."""
    import lumilake_server.routes.jobs as jobs_module
    from lumilake_server.hooks import USAGE_SINKS

    saved_statuses: list[str] = []
    real_save = jobs_module._job_storage.save
    target: dict[str, str] = {}

    def _spy_save(record: Any) -> None:
        if target and record.job_id == target["id"]:
            saved_statuses.append(record.status)
        real_save(record)

    monkeypatch.setattr(jobs_module._job_storage, "save", _spy_save)

    class _UsageSpy:
        name = "test.usage_spy"
        calls = 0

        async def emit(self, rows: Any, logger: Any) -> None:
            _UsageSpy.calls += 1

    USAGE_SINKS.append(_UsageSpy())

    from lumilake_server.hooks import ResourceKind

    trace_regs: list[str] = []
    real_register = jobs_module.register_resource

    async def _spy_register(
        principal: Any, kind: Any, rid: Any, meta: Any, logger: Any
    ) -> None:
        if kind == ResourceKind.TRACE:
            trace_regs.append(rid)
        await real_register(principal, kind, rid, meta, logger)

    monkeypatch.setattr(jobs_module, "register_resource", _spy_register)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    target["id"] = job_id
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [{"next": "bogus"}]
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "failed"
    assert (
        "failed" in saved_statuses
    ), f"failed terminal status never durably saved; saw {saved_statuses!r}"
    assert (
        _UsageSpy.calls == 1
    ), f"terminal hooks fired {_UsageSpy.calls} times, expected exactly once"
    # Trace resources are registered once per id in the parent's aggregated
    # trace_ids (collected from child records + the runtime), not per child job.
    assert (
        trace_regs == record.trace_ids
    ), f"failed path registered {trace_regs}, parent aggregated {record.trace_ids}"
    assert record.trace_ids, "failed path aggregated no trace ids; check is vacuous"


@pytest.mark.anyio
async def test_loop_cancelled_terminal_persisted_and_hooks_fire_once(
    app: Any,
    job_routes: Any,
    monkeypatch: pytest.MonkeyPatch,
    wait_for_inflight_child: Any,
) -> None:
    """A CANCELLED parent's terminal status is durably saved and terminal hooks
    fire exactly once."""
    import lumilake_server.routes.jobs as jobs_module
    from lumilake_server.hooks import USAGE_SINKS

    saved_statuses: list[str] = []
    real_save = jobs_module._job_storage.save
    target: dict[str, str] = {}

    def _spy_save(record: Any) -> None:
        if target and record.job_id == target["id"]:
            saved_statuses.append(record.status)
        real_save(record)

    monkeypatch.setattr(jobs_module._job_storage, "save", _spy_save)

    class _UsageSpy:
        name = "test.usage_spy"
        calls = 0

        async def emit(self, rows: Any, logger: Any) -> None:
            _UsageSpy.calls += 1

    USAGE_SINKS.append(_UsageSpy())

    from lumilake_server.hooks import ResourceKind

    trace_regs: list[str] = []
    real_register = jobs_module.register_resource

    async def _spy_register(
        principal: Any, kind: Any, rid: Any, meta: Any, logger: Any
    ) -> None:
        if kind == ResourceKind.TRACE:
            trace_regs.append(rid)
        await real_register(principal, kind, rid, meta, logger)

    monkeypatch.setattr(jobs_module, "register_resource", _spy_register)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    target["id"] = job_id
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    fake_server.hang_rounds = {1}
    fake_server.cancel_raises_cancelled = True
    loop_task = asyncio.create_task(_run_background(app))
    child_id = await wait_for_inflight_child(job_routes, job_id)
    job_routes.jobs[child_id].status = "cancelled"
    # A cancelled child that DID reach the runtime carries trace ids; seed them
    # so the parent aggregates them into its terminal trace resources.
    job_routes.jobs[child_id].trace_ids = ["trace-cancelled-0"]
    fake_server.hang_event.set()
    await loop_task
    record = job_routes.jobs[job_id]
    assert record.status == "cancelled"
    assert (
        "cancelled" in saved_statuses
    ), f"cancelled terminal status never durably saved; saw {saved_statuses!r}"
    assert (
        _UsageSpy.calls == 1
    ), f"terminal hooks fired {_UsageSpy.calls} times, expected exactly once"
    # Trace resources are registered once per id in the parent's aggregated
    # trace_ids (collected from child records + the runtime), not per child job.
    assert (
        trace_regs == record.trace_ids
    ), f"cancelled path registered {trace_regs}, parent aggregated {record.trace_ids}"
    assert record.trace_ids, "cancelled path aggregated no trace ids; check is vacuous"


@pytest.mark.anyio
async def test_cancel_job_with_failing_backend_cancel_marks_child_failed(
    app: Any, job_routes: Any, wait_for_inflight_child: Any
) -> None:
    """Cancelling a dynamic parent through the real endpoint with a failing
    backend cancel must record the in-flight child failed while the parent
    stays cancelled."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    fake_server.hang_rounds = {1}
    fake_server.fail_cancel = True
    loop_task = asyncio.create_task(_run_background(app))
    # Wait for the round-1 child (the one that hangs) to be genuinely in flight.
    child_id = await wait_for_inflight_child(job_routes, job_id)
    # Cancel through the real endpoint.
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        cancel_resp = await client.post(
            f"/jobs/{job_id}/cancel",
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert cancel_resp.status_code == 200, cancel_resp.text
    await loop_task
    assert job_routes.jobs[job_id].status == "cancelled"
    child = job_routes.jobs[child_id]
    assert child.status == "failed", f"expected failed, got {child.status!r}"
    assert child.error == "cancellation failed: cancel RPC failed", child.error
    # The backend cancel RPC must have been issued for the in-flight child.
    assert (
        child_id in fake_server.cancel_calls
    ), f"child backend cancel RPC not issued; cancel_calls={fake_server.cancel_calls}"


@pytest.mark.anyio
async def test_round_timeout_with_failing_backend_cancel_marks_child_failed(
    app: Any, job_routes: Any
) -> None:
    """A round that exceeds job_timeout with a failing backend cancel must
    record the child failed and fail the parent."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_TIMEOUT_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    # Round 0's child hangs past the 0.05s job_timeout; the backend cancel then
    # fails. cancel_request sets hang_event before raising, so the run
    # terminates on its own — do not set hang_event manually.
    fake_server.hang_rounds = {0}
    fake_server.fail_cancel = True
    await _run_background(app)
    parent = job_routes.jobs[job_id]
    assert parent.status == "failed"
    child_id = parent.child_job_ids[-1]
    assert parent.error == (
        f"dynamic round 0 ({child_id}) exceeded job_timeout 0.05s "
        "and backend cancellation failed"
    )
    child = job_routes.jobs[child_id]
    assert child.status == "failed"
    assert child.error == "cancellation failed after timeout: cancel RPC failed"
    assert child_id in fake_server.cancel_calls


# ---------------------------------------------------------------------------
# CHECK 8: library admission / prompt / resolution domains agree
# ---------------------------------------------------------------------------
# Three places each decide what a library entry may be: validate_library
# (admission), system_message (prompt), and resolve_subgraph (resolution).
# Nothing asserts they agree. This section pins root cause C by deriving the
# fixture from the real SUPPORTED_OPS constant so adding an op type to the
# runtime breaks the suite until a template is added.

_LIBRARY_TEMPLATES: dict[str, dict[str, Any]] = {
    "DataOp": {"op": "DataOp", "data": [1, 2, 3]},
    "DataRetrievalOp": {
        "op": "DataRetrievalOp",
        "data_spec": {
            "type": "lumid",
            "mode": "sql",
            "output_format": "jsonl",
            "verify": False,
            "template": "SELECT 1",
        },
    },
    "EmbeddingOp": {"op": "EmbeddingOp", "model": "m"},
    "FormatOp": {"op": "FormatOp", "template": "{x}"},
    "ImageGenerationOp": {"op": "ImageGenerationOp", "model": "m"},
    "LLMChatOp": {"op": "LLMChatOp", "model": "m"},
    "LLMVisionOp": {"op": "LLMVisionOp", "model": "m"},
    "LambdaOp": {"op": "LambdaOp", "code": "def f(x):\n    return x\n"},
    "MessageOp": {"op": "MessageOp", "content": "hi"},
}


def test_library_template_fixture_covers_every_op_type() -> None:
    """The fixture's key set must equal SUPPORTED_OPS so a new op type breaks
    this suite until a template is added."""
    assert set(_LIBRARY_TEMPLATES) == SUPPORTED_OPS, (
        f"library template fixture diverges from SUPPORTED_OPS; symmetric "
        f"difference: {sorted(set(_LIBRARY_TEMPLATES) ^ set(SUPPORTED_OPS))}"
    )


@pytest.mark.parametrize("op_type", sorted(SUPPORTED_OPS))
def test_library_admission_prompt_and_resolution_agree(op_type: str) -> None:
    """The admission, prompt, and resolution domains must all accept the same
    library entry for every op type."""
    lib: dict[str, dict[str, Any]] = {f"tpl_{op_type}": _LIBRARY_TEMPLATES[op_type]}
    ref = f"tpl_{op_type}"
    # a. ADMISSION domain: validate_library must accept the entry.
    validate_library(lib)
    # b. PROMPT domain: the ref id must appear in the rendered system message.
    msg = system_message("goal", [], [], None, lib)
    assert ref in msg, (
        f"library entry {ref!r} (op {op_type!r}) missing from the system "
        "message prompt"
    )
    # c. RESOLUTION domain: resolve_subgraph must expand the ref to the op type.
    resolved = resolve_subgraph([{"ref": ref, "id": "n1", "inputs": []}], lib)
    assert (
        len(resolved) == 1
    ), f"library entry {ref!r} (op {op_type!r}) did not resolve to one op"
    assert resolved[0]["op"] == op_type, (
        f"library entry {ref!r} resolved to {resolved[0]['op']!r}, expected "
        f"{op_type!r}"
    )


def test_library_admission_and_resolution_reject_the_same_unknown_ref() -> None:
    """Both the admission and resolution doors must reject an unknown entry."""
    with pytest.raises(DriverProtocolError):
        validate_library({"bad": {"op": "NotAnOp"}})
    with pytest.raises(DriverProtocolError):
        resolve_subgraph([{"ref": "missing", "id": "n1", "inputs": []}], {})


@pytest.mark.anyio
async def test_runtime_reported_cancellation_marks_record_cancelled(
    app: Any, job_routes: Any
) -> None:
    """A runtime-reported cancellation (not initiated by the route) must leave
    the child record cancelled, not a zombie running record."""
    import lumilake_server.routes.jobs as jobs_module

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    fake_server.hang_rounds = {0}
    fake_server.cancel_raises_cancelled = True
    # Set the hang event BEFORE running the loop so round 0's execute raises
    # RequestCancelledError immediately while the child record is still running.
    fake_server.hang_event.set()
    await _run_background(app)
    parent = job_routes.jobs[job_id]
    child_id = parent.child_job_ids[-1]
    child = job_routes.jobs[child_id]
    assert child.status == "cancelled", (
        f"runtime-reported cancellation must mark the child cancelled, "
        f"got {child.status!r}"
    )
    assert (
        child.finished_at is not None
    ), "runtime-reported cancellation must set finished_at"
    # The persisted record must agree, not just the in-memory one.
    persisted = jobs_module._job_storage.load(child_id)
    assert persisted is not None, f"child {child_id} not found in storage"
    assert (
        persisted["status"] == "cancelled"
    ), f"persisted child status must be cancelled, got {persisted['status']!r}"


@pytest.mark.anyio
async def test_terminal_parent_survives_failed_child_path(
    app: Any, job_routes: Any, wait_for_inflight_child: Any
) -> None:
    """A parent marked failed (e.g. by shutdown recovery) while a child is in
    flight must not be resurrected to completed by the dynamic loop's
    failed-child branch."""
    import lumilake_server.routes.jobs as jobs_module

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    # Round 0 hangs so a child is genuinely in flight when shutdown lands.
    fake_server.hang_rounds = {0}
    loop_task = asyncio.create_task(_run_background(app))
    # Wait until the child is genuinely executing (blocked on hang_event), not
    # merely registered, so shutdown lands while it is truly in flight.
    await wait_for_inflight_child(job_routes, job_id)
    # Mark the parent failed while the child is in flight, as shutdown recovery
    # would.
    await jobs_module.mark_running_jobs_failed("server shutdown")
    # Release the child so the loop can finish.
    fake_server.hang_event.set()
    await loop_task
    parent = job_routes.jobs[job_id]
    assert (
        parent.status == "failed"
    ), f"parent must stay failed after the loop, got {parent.status!r}"
    assert (
        parent.error == "server shutdown"
    ), f"parent error must be preserved, got {parent.error!r}"
    assert parent.status != "completed", "parent must not be resurrected to completed"
    assert parent.status != "running", "parent must not be left running"
    assert fake_server.execute_calls, (
        "child never executed; the failed-child terminal-authority path was "
        "not exercised"
    )
    persisted = jobs_module._job_storage.load(job_id)
    assert persisted is not None, f"parent {job_id} not found in storage"
    assert (
        persisted["status"] == "failed"
    ), f"persisted parent status must be failed, got {persisted['status']!r}"
    assert (
        persisted["error"] == "server shutdown"
    ), f"persisted parent error must be preserved, got {persisted['error']!r}"


@pytest.mark.anyio
async def test_terminal_parent_survives_normal_completion_path(
    app: Any, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent marked failed (e.g. by shutdown recovery) must not be
    resurrected to completed by the dynamic loop's normal-completion guard."""
    import lumilake_server.routes.jobs as jobs_module

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    # Round 0 stops directly so the loop breaks straight to the completion
    # guard; a second round would be refused once the parent is terminal.
    fake_server.plans = [{"next": "STOP"}]

    real_submit_child = jobs_module._submit_dynamic_child
    marked = {"done": False}

    async def _spy_submit_child(*args: Any, **kwargs: Any) -> Any:
        child_id = await real_submit_child(*args, **kwargs)
        if not marked["done"]:
            marked["done"] = True
            parent_id = kwargs["parent_job_id"]
            async with jobs_module.jobs_lock:
                parent = jobs_module.jobs[parent_id]
                parent.status = "failed"
                parent.error = "server shutdown"
                parent.finished_at = jobs_module._now()
                snapshot = parent
            await asyncio.to_thread(jobs_module._job_storage.save, snapshot)
        return child_id

    monkeypatch.setattr(jobs_module, "_submit_dynamic_child", _spy_submit_child)

    await _run_background(app)
    parent = job_routes.jobs[job_id]
    assert (
        len(parent.child_job_ids) == 1
    ), f"expected exactly one round; got {parent.child_job_ids}"
    child_id = parent.child_job_ids[-1]
    child = job_routes.jobs[child_id]
    assert child.status == "completed", (
        f"child must complete so the loop takes the normal-completion path, "
        f"got {child.status!r}"
    )
    assert (
        parent.status == "failed"
    ), f"parent must stay failed after the loop, got {parent.status!r}"
    assert (
        parent.error == "server shutdown"
    ), f"parent error must be preserved, got {parent.error!r}"
    assert parent.status != "completed", "parent must not be resurrected to completed"
    persisted = jobs_module._job_storage.load(job_id)
    assert persisted is not None, f"parent {job_id} not found in storage"
    assert (
        persisted["status"] == "failed"
    ), f"persisted parent status must be failed, got {persisted['status']!r}"
    assert (
        persisted["error"] == "server shutdown"
    ), f"persisted parent error must be preserved, got {persisted['error']!r}"


def test_no_inline_terminal_status_literal_in_jobs_module() -> None:
    """The terminal status set must be referenced from TERMINAL_JOB_STATUSES,
    never written inline in any collection form (set, tuple, list or
    frozenset(...)). Exactly one such literal is permitted: the one inside the
    constant's own assignment."""
    jobs_source = (
        _REPO_ROOT / "src" / "lumilake_server" / "routes" / "jobs.py"
    ).read_text()
    tree = ast.parse(jobs_source)

    terminal = frozenset({"completed", "failed", "cancelled"})
    offending: list[int] = []
    constant_start: int | None = None
    constant_end: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            assignment: ast.Assign | ast.AnnAssign | None = node
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            assignment = node
        else:
            targets = []
            assignment = None
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "TERMINAL_JOB_STATUSES":
                assert assignment is not None
                constant_start = assignment.lineno
                constant_end = assignment.end_lineno
        if isinstance(node, (ast.Set, ast.Tuple, ast.List)) and node.elts:
            values = [
                el.value
                for el in node.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            ]
            if len(values) == len(node.elts) and set(values) == terminal:
                offending.append(node.lineno)

    assert (
        constant_start is not None and constant_end is not None
    ), "TERMINAL_JOB_STATUSES assignment not found in jobs.py"
    assert len(offending) == 1, (
        f"terminal status set written inline at lines {offending}; "
        "reference TERMINAL_JOB_STATUSES instead"
    )
    assert constant_start <= offending[0] <= constant_end, (
        f"the only terminal status literal is at line {offending[0]}, but "
        f"TERMINAL_JOB_STATUSES spans lines {constant_start}-{constant_end}"
    )


@pytest.mark.parametrize("op_type", sorted(SUPPORTED_OPS))
def test_output_source_ops_matches_runtime_predicate(op_type: str) -> None:
    """The driver's admitted OutputOp source set must match the runtime's
    predicate (an LLMOp subclass or DataRetrievalOp)."""
    from lumilake_server import ops as ops_pkg
    from lumilake_server.dynamic.driver import _OUTPUT_SOURCE_OPS
    from lumilake_server.ops.data_ops import DataRetrievalOp
    from lumilake_server.ops.llm_ops import LLMOp

    cls = vars(ops_pkg)[op_type]
    runtime_admits = issubclass(cls, (LLMOp, DataRetrievalOp))
    driver_admits = op_type in _OUTPUT_SOURCE_OPS
    assert driver_admits == runtime_admits, (
        f"op {op_type!r}: driver admits={driver_admits}, runtime admits="
        f"{runtime_admits}"
    )


def test_embedding_leaf_is_admitted_end_to_end() -> None:
    """An EmbeddingOp leaf must pass validation and build_round, proving the
    widened membership is used, not merely computed."""
    from lumilake_server.dynamic.driver import build_round, validate_emitted_subgraph

    subgraph: list[dict[str, Any]] = [
        {
            "id": "m",
            "op": "MessageOp",
            "inputs": [],
            "messages": [{"role": "user", "content": "hi"}],
        },
        {
            "id": "e",
            "op": "EmbeddingOp",
            "inputs": ["m"],
            "config": {"model": "Qwen/Qwen3-8B"},
            "input": "m",
        },
    ]
    validate_emitted_subgraph(subgraph, {}, 8)
    round_build = build_round(
        subgraph,
        node_registry={},
        round_index=1,
        goal="g",
        observations=[],
        topology=[],
        preview_width=900,
        model="Qwen/Qwen3-8B",
        max_tokens=768,
        temperature=0.4,
    )
    assert any(
        "embedding" in name for name in round_build.leaf_output_names
    ), f"expected an embedding leaf output, got {round_build.leaf_output_names}"


@pytest.mark.anyio
async def test_round0_render_matches_loop_build(
    app: Any, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The round-0 graph rendered at submit/preview time must equal the one the
    loop builds at runtime, so the previewed graph is the graph that runs."""
    import lumilake_server.routes.jobs as jobs_module

    captures: list[tuple[int, Any]] = []
    real_build_round = jobs_module.build_round

    def _spy(*args: Any, **kwargs: Any) -> Any:
        result = real_build_round(*args, **kwargs)
        captures.append((kwargs["round_index"], result.graph))
        return result

    monkeypatch.setattr(jobs_module, "build_round", _spy)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_ROUND0_DIFF_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    await _run_background(app)

    round0 = [graph for idx, graph in captures if idx == 0]
    assert len(round0) == 2, (
        f"expected exactly two round-0 build_round calls (submit render + loop), "
        f"got {len(round0)}"
    )
    render_graph, loop_graph = round0

    # Detail the mismatch by node id: symmetric difference of keys plus keys
    # whose values differ.
    render_keys = set(render_graph)
    loop_keys = set(loop_graph)
    differing: set[str] = set()
    for key in render_keys & loop_keys:
        if render_graph[key] != loop_graph[key]:
            differing.add(key)
    assert render_graph == loop_graph, (
        f"round-0 render and loop graphs differ; keys only in render: "
        f"{sorted(render_keys - loop_keys)}, only in loop: "
        f"{sorted(loop_keys - render_keys)}, differing values: {sorted(differing)}"
    )

    # Independently assert the captured round-0 graph carries the non-default
    # settings, so a drift that drops the same field on BOTH paths still fails.
    proposer = render_graph.get("proposer", {})
    config = proposer.get("config", {})
    assert (
        proposer["config"]["model"] == "Qwen/Qwen3-32B"
    ), f"proposer model {proposer['config']['model']!r} is not the spec's driver.model"
    assert config.get("max_tokens") == 1234, (
        "round-0 proposer max_tokens not carried through, got "
        f"{config.get('max_tokens')!r}"
    )
    assert config.get("temperature") == 0.9, (
        "round-0 proposer temperature not carried through, got "
        f"{config.get('temperature')!r}"
    )
    message = render_graph.get("message", {})
    system_messages = message.get("messages", [])
    system = ""
    for entry in system_messages:
        if isinstance(entry, dict) and entry.get("role") == "system":
            system = entry.get("content", "")
    assert "0.7" in system, "round-0 system message does not mention threshold 0.7"
    assert (
        "sector_market_cap" in system
    ), "round-0 system message does not mention the library entry"


# Op types the internal-id tripwire parametrization exercises, derived from the
# real constant so a new op type is picked up automatically.
_INTERNAL_ID_OP_TYPES = sorted(SUPPORTED_OPS)


@pytest.mark.parametrize("op_type", _INTERNAL_ID_OP_TYPES)
def test_internal_id_matches_real_parser_output(op_type: str) -> None:
    """The derived internal id must match the id the real YAML parser assigns,
    so a drift in the parser's id derivation is caught."""
    op = _valid_op(op_type, "a")
    user_id = op["id"]
    workflow = {
        "name": "depcheck",
        "inputs": {"Symbols": []},
        "ops": [_stub("a"), op, _consumer()],
        "outputs": [],
    }
    parsed = parse_yaml_payload(workflow)
    graph_name = next(iter(parsed))
    native = parsed[graph_name]["graph"]
    derived = _internal_id(graph_name, op_type, user_id)
    assert derived in native, (
        f"op {op_type!r}: derived internal id {derived!r} not in parsed graph; "
        f"actual keys: {sorted(native)}"
    )


def test_internal_id_fixture_covers_every_op_type() -> None:
    """The parametrization must exercise every SUPPORTED_OPS op type."""
    assert set(_INTERNAL_ID_OP_TYPES) == SUPPORTED_OPS, (
        f"internal-id tripwire diverges from SUPPORTED_OPS; symmetric "
        f"difference: {sorted(set(_INTERNAL_ID_OP_TYPES) ^ set(SUPPORTED_OPS))}"
    )


_RENDERER_REJECT_ENVELOPES = [
    # Non-default poll_interval.
    {
        "data": [
            {
                "name": "demo",
                "workflow": _VALID_DYNAMIC_YAML.replace(
                    "  model: Qwen/Qwen3-8B\n",
                    "  model: Qwen/Qwen3-8B\n  poll_interval: 5.0\n",
                ),
                "inputs": {"Symbols": ["NVDA"]},
                "output_location": {"type": "s3", "prefix": "dynamic/data-free/"},
            }
        ]
    },
    # Library entry with an unsupported op type.
    {
        "data": [
            {
                "name": "demo",
                "workflow": _VALID_DYNAMIC_YAML + "library:\n  bad:\n    op: NotAnOp\n",
                "inputs": {"Symbols": ["NVDA"]},
                "output_location": {"type": "s3", "prefix": "dynamic/data-free/"},
            }
        ]
    },
]


@pytest.mark.anyio
async def test_renderer_rejections_reach_both_doors(
    app: Any, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared renderer must be the thing rejecting for both doors."""
    import lumilake_server.routes.jobs as jobs_module

    calls = {"n": 0}
    real_render = jobs_module._render_dynamic_round0

    def _spy(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real_render(*args, **kwargs)

    monkeypatch.setattr(jobs_module, "_render_dynamic_round0", _spy)

    transport = httpx.ASGITransport(app=app)
    for envelope in _RENDERER_REJECT_ENVELOPES:
        calls["n"] = 0
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            submit_resp = await client.post(
                "/jobs",
                json=envelope,
                headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
            )
            preview_resp = await client.post(
                "/jobs/preview",
                json=envelope,
                headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
            )
        assert submit_resp.status_code == 422, submit_resp.text
        assert preview_resp.status_code == 422, preview_resp.text
        assert (
            calls["n"] == 2
        ), f"renderer entered {calls['n']} times, expected 2 (one per door)"


@pytest.mark.anyio
async def test_bare_string_library_entry_rejected_before_renderer(
    app: Any, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare-string library entry is a pydantic decode failure, rejected before
    the renderer is entered."""
    from pydantic import ValidationError

    import lumilake_server.routes.jobs as jobs_module
    from lumilake_server.dynamic.spec import DynamicSpec

    calls = {"n": 0}
    real_render = jobs_module._render_dynamic_round0

    def _spy(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real_render(*args, **kwargs)

    monkeypatch.setattr(jobs_module, "_render_dynamic_round0", _spy)

    bare_yaml = _VALID_DYNAMIC_YAML + "library:\n  bad: nope\n"
    envelope = {
        "data": [
            {
                "name": "demo",
                "workflow": bare_yaml,
                "inputs": {"Symbols": ["NVDA"]},
            }
        ]
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        submit_resp = await client.post(
            "/jobs",
            json=envelope,
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
        preview_resp = await client.post(
            "/jobs/preview",
            json=envelope,
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert submit_resp.status_code == 422, submit_resp.text
    assert preview_resp.status_code == 422, preview_resp.text
    assert calls["n"] == 0, (
        f"renderer entered {calls['n']} times, expected 0 (pydantic decode rejects "
        "before the renderer)"
    )
    with pytest.raises(ValidationError):
        DynamicSpec.model_validate(
            {
                "name": "dynamic",
                "goal": "analyze market data",
                "driver": {"model": "Qwen/Qwen3-8B"},
                "library": {"bad": "nope"},
            }
        )
