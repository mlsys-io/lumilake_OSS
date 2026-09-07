"""Validation and graph-assembly helpers for dynamic workflow rounds.

A round is a single job whose DAG is ``[emitted subgraph] -> [observation
LambdaOp] -> [proposer LLMChatOp]``. The proposer reads the goal, all prior
observations, and the overall topology, and emits a structured plan selecting
the next subgraph (a small acyclic DAG of ops) or ``STOP``. These helpers
validate the emitted subgraph, assemble the fused round graph, and validate
the returned plan.
"""

import ast
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from lumilake_server import ops as ops_pkg
from lumilake_server.dynamic.blocks import (
    INPUT_NODE_ID,
    OBSERVATION_NODE_ID,
    PROPOSER_NODE_ID,
    fused_round_graph,
)
from lumilake_server.ops.data_ops import DataRetrievalOp
from lumilake_server.ops.llm_ops import LLMOp
from lumilake_server.parser.common import make_id
from lumilake_server.parser.yaml_parser import (
    SUPPORTED_OPS,
    _op_id_prefix,
    parse_yaml_payload,
)

_OBSERVATION_TEMPLATE_FILE = Path(__file__).with_name("observation_template.py")


def observation_lambda(*, preview_width: int = 900) -> str:
    """Load the observation LambdaOp source for ``preview_width`` chars.

    Reads the standalone ``observation_template.py`` and substitutes the
    preview character budget into its ``{width}`` placeholder.
    """
    source = _OBSERVATION_TEMPLATE_FILE.read_text()
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise DriverProtocolError(
            f"observation template is not valid Python: {exc}"
        ) from exc
    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "observe"
    ]
    if len(functions) != 1:
        raise DriverProtocolError(
            "observation template must define exactly one top-level observe function"
        )
    target = functions[0]
    start_line = target.lineno
    if target.decorator_list:
        start_line = min(decorator.lineno for decorator in target.decorator_list)
    lines = source.splitlines(keepends=True)
    function_source = "".join(lines[start_line - 1 : target.end_lineno])
    return function_source.replace("{width}", str(preview_width))


STOP = "STOP"
SUBGRAPH = "subgraph"

# Round-graph node ids a subgraph node id must not collide with.
FIXED_ROUND_NODE_IDS: frozenset[str] = frozenset(
    {
        INPUT_NODE_ID,
        OBSERVATION_NODE_ID,
        PROPOSER_NODE_ID,
    }
)

# Op types the runtime permits as an OutputOp source; every subgraph leaf is
# wrapped in an OutputOp, so leaves must be one of these. Derived from the
# runtime's admission predicate so the driver is never stricter than the
# runtime.
_OP_CLASSES: dict[str, type] = {
    name: obj for name, obj in vars(ops_pkg).items() if isinstance(obj, type)
}
_MISSING_OP_CLASSES = SUPPORTED_OPS - _OP_CLASSES.keys()
if _MISSING_OP_CLASSES:
    raise RuntimeError(
        f"op types not exported by lumilake_server.ops: {sorted(_MISSING_OP_CLASSES)}"
    )
_OUTPUT_SOURCE_OPS: frozenset[str] = frozenset(
    name
    for name in SUPPORTED_OPS
    if issubclass(_OP_CLASSES[name], (LLMOp, DataRetrievalOp))
)


class DriverProtocolError(Exception):
    """Raised when a job result or driver configuration violates the contract.

    Distinct from :class:`RuntimeError` so callers can distinguish a malformed
    plan/config from a job that legitimately failed.
    """


class StopPlan(BaseModel):
    """Validated plan indicating the loop should halt with ``STOP``."""

    model_config = ConfigDict(extra="forbid")


class SubgraphPlan(BaseModel):
    """Validated plan selecting the next round's subgraph."""

    model_config = ConfigDict(extra="forbid")

    ops: list[dict[str, Any]]


class _RawPlan(BaseModel):
    """Structural shape of an untrusted plan dict."""

    model_config = ConfigDict(extra="forbid")

    next: str
    ops: list[dict[str, Any]] | None = None


class RoundBuild(BaseModel):
    """Typed result of assembling one round.

    ``graph`` is the native round graph; ``leaf_output_names`` lists the
    archived ``leaf_<internal_id>`` output names in deterministic order, so the
    route can validate that every expected leaf produced a result.
    """

    model_config = ConfigDict(extra="forbid")

    graph: dict[str, Any]
    leaf_output_names: list[str]


def round_output_location(
    base: dict[str, Any], run_namespace: str, round_index: int
) -> dict[str, Any]:
    """Derive a distinct output location for one round within a run.

    Folder outputs use fixed item names, so reusing one prefix would overwrite
    the previous round's export.
    """
    location_type = base["type"]
    if location_type == "s3":
        prefix = base["prefix"]
        sep = "" if prefix.endswith("/") else "/"
        return {
            "type": "s3",
            "prefix": f"{prefix}{sep}{run_namespace}/round-{round_index}/",
        }
    raise DriverProtocolError(f"unsupported output location type {location_type!r}")


def validate_library(library: dict[str, dict[str, Any]] | None) -> None:
    """Validate the reference library of pre-configured op templates.

    Each entry must be a mapping whose ``op`` type is a known op type. Raises
    :class:`DriverProtocolError` on any violation.
    """
    if not library:
        return
    for ref, template in library.items():
        if not isinstance(template, dict):
            raise DriverProtocolError(f"library entry {ref!r} must be a mapping")
        op_type = template.get("op")
        if not isinstance(op_type, str) or op_type not in SUPPORTED_OPS:
            raise DriverProtocolError(
                f"library entry {ref!r} has unsupported op type {op_type!r}; "
                f"supported: {sorted(SUPPORTED_OPS)}"
            )


# Fields a planner may override when referencing a library template. The
# template is authoritative for its config; only identity and wiring vary.
# ``op`` is a matching assertion (checked, not stored); ``inputs`` wires the
# template's declared param slots.
_LIBRARY_REF_OVERRIDES = frozenset({"id", "inputs", "op"})


def _apply_param_bindings(
    template: dict[str, Any], ref: str, inputs: list[Any]
) -> dict[str, Any]:
    """Substitute the planner's ``inputs`` into the template's param slots.

    A library retrieval template declares ``param_bindings``: a list of
    ``{"label": <param label>, "input": <index into inputs>}``. Each binding
    sets the named param's ``node`` to the corresponding input, so the planner
    wires a retrieval to a prior-round node purely through ``inputs`` without
    mutating the template's config. The input count must match the declared
    binding count exactly, and every binding label must exist in the template's
    params.
    """
    data_spec = template.get("data_spec")
    if not isinstance(data_spec, dict):
        raise DriverProtocolError(
            f"library entry {ref!r} is not a retrieval op; only retrieval ops "
            "declare param bindings"
        )
    bindings = data_spec.get("param_bindings")
    if not isinstance(bindings, list):
        # No declared bindings: inputs are used only for closure/leaf wiring,
        # not native param substitution. Leave the config untouched.
        return data_spec
    if len(inputs) != len(bindings):
        raise DriverProtocolError(
            f"library entry {ref!r} declares {len(bindings)} param binding(s) "
            f"but got {len(inputs)} input(s)"
        )
    params = data_spec.get("params")
    if not isinstance(params, list):
        raise DriverProtocolError(
            f"library entry {ref!r} has param_bindings but no params"
        )
    by_label = {param.get("label"): dict(param) for param in params}
    for binding in bindings:
        if not isinstance(binding, dict):
            raise DriverProtocolError(
                f"library entry {ref!r} param_bindings entries must be mappings"
            )
        label = binding.get("label")
        slot = binding.get("input")
        if label not in by_label:
            raise DriverProtocolError(
                f"library entry {ref!r} param_binding references unknown "
                f"label {label!r}; params: {sorted(by_label)}"
            )
        if not isinstance(slot, int) or not 0 <= slot < len(inputs):
            raise DriverProtocolError(
                f"library entry {ref!r} param_binding for {label!r} has "
                f"invalid input slot {slot!r}"
            )
        by_label[label]["node"] = inputs[slot]
    return {**data_spec, "params": list(by_label.values())}


def resolve_subgraph(
    subgraph: list[dict[str, Any]],
    library: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Expand library references in an emitted subgraph into full op configs.

    An op that carries a ``ref`` key names a library template; the resolved op
    is the template's config with the emitted op's ``id`` and ``inputs`` applied
    on top. ``inputs`` wires the template's declared param slots (see
    :func:`_apply_param_bindings`); a declared ``op`` is asserted to match the
    template and ignored during merge. The planner may not override the
    template's other config (``data_spec``, ``model``, etc.). Ops without a
    ``ref`` pass through unchanged (fully-inline ops are allowed), so this is
    idempotent.
    """
    resolved: list[dict[str, Any]] = []
    for op in subgraph:
        if "ref" not in op:
            resolved.append(op)
            continue
        ref = op["ref"]
        if not isinstance(ref, str):
            raise DriverProtocolError(
                f"subgraph op {op.get('id')!r} 'ref' must be a string"
            )
        if not library or ref not in library:
            raise DriverProtocolError(
                f"subgraph op {op.get('id')!r} references unknown library "
                f"entry {ref!r}"
            )
        template = library[ref]
        merged = dict(template)
        declared_op = op.get("op")
        if declared_op is not None and merged.get("op") != declared_op:
            raise DriverProtocolError(
                f"subgraph op {op.get('id')!r} declares op {declared_op!r} "
                f"but library entry {ref!r} is {merged.get('op')!r}"
            )
        for key, value in op.items():
            if key == "ref" or key == "op":
                continue
            if key not in _LIBRARY_REF_OVERRIDES:
                raise DriverProtocolError(
                    f"subgraph op {op.get('id')!r} may not override "
                    f"{key!r} when referencing library entry {ref!r}; only "
                    "id and inputs may vary"
                )
            if key == "inputs":
                if not isinstance(value, list) or not all(
                    isinstance(v, str) for v in value
                ):
                    raise DriverProtocolError(
                        f"subgraph op {op.get('id')!r} inputs must be a list "
                        "of node ids"
                    )
                merged["inputs"] = value
                # A non-retrieval template (no data_spec) with no inputs has no
                # param slots to wire; resolve it as-is. Retrieval templates
                # wire their declared param slots.
                if isinstance(template.get("data_spec"), dict):
                    merged["data_spec"] = _apply_param_bindings(template, ref, value)
                continue
            merged[key] = value
        resolved.append(merged)
    return resolved


def _validate_declared_inputs(
    op_id: str,
    op_type: str,
    op: dict[str, Any],
    inputs: list[Any],
    available_ops: dict[str, dict[str, Any]],
) -> None:
    """Reject a mismatch between declared inputs and the native dependency set.

    The parser is the source of truth for what an op consumes. We build the op
    through the same ``parse_yaml_payload`` path ``build_round`` uses, then
    compare the declared ``inputs`` set against the parser-derived native
    dependency set in BOTH directions: a declared input the native config does
    not consume is rejected, and a native reference missing from declared
    inputs is rejected. If the op config references an undeclared node, the
    parser raises ``unknown id``, which we surface as a native-but-undeclared
    rejection.
    """
    declared = {ref for ref in inputs if ref != INPUT_NODE_ID}
    # Build the probe from the REAL referenced ops so type-constrained
    # references (e.g. LLMChatOp messages_ref -> MessageOp) resolve. Include
    # each declared ref's actual op plus whatever it transitively references;
    # fall back to a DataOp stub only for a ref absent from available_ops.
    probe_ops: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_probe(ref: str) -> None:
        if ref == INPUT_NODE_ID or ref == op_id or ref in seen:
            return
        seen.add(ref)
        real = available_ops.get(ref)
        if real is None:
            probe_ops.append({"id": ref, "op": "DataOp", "inputs": [], "data": ["x"]})
            return
        # Drop any reference back to the op being validated: in a valid
        # (acyclic) subgraph no referenced op points at it, so this only
        # affects cyclic subgraphs, which the acyclic check rejects anyway.
        probe_op = dict(real)
        probe_op["inputs"] = [dep for dep in real.get("inputs", []) if dep != op_id]
        probe_ops.append(probe_op)
        for dep in probe_op["inputs"]:
            if isinstance(dep, str):
                add_probe(dep)

    for ref in declared:
        add_probe(ref)
    workflow = {
        "name": "depcheck",
        "inputs": {INPUT_NODE_ID: []},
        "ops": [op, *probe_ops],
        "outputs": [],
    }
    try:
        parsed = parse_yaml_payload(workflow)
    except ValueError as exc:
        # The parser could not resolve a config reference not in declared
        # inputs — a native-but-undeclared dependency.
        raise DriverProtocolError(
            f"subgraph op {op_id!r} references a node not in its declared "
            f"inputs: {exc}"
        ) from exc
    graph_name = next(iter(parsed))
    spec = parsed[graph_name]
    native = spec["graph"]
    # Build the reverse internal->user map LOCALLY from the probe ops we
    # authored, using the same id derivation the parser uses. Implicit nodes
    # the parser synthesises (MessageOp/FormatOp) are not in this map.
    internal_to_user: dict[str, str] = {}
    for probe_op in [op, *probe_ops]:
        uid = probe_op.get("id")
        otype = probe_op.get("op")
        if isinstance(uid, str) and isinstance(otype, str):
            internal_to_user[_internal_id(graph_name, otype, uid)] = uid

    native_op = native.get(_internal_id(graph_name, op_type, op_id))
    if native_op is None:
        raise DriverProtocolError(
            f"subgraph op {op_id!r} did not produce a native node"
        )

    # Derive the native dependency set from the parsed graph: collect every
    # string in the op's config fields that is an internal id, then walk
    # ``_inputs`` transitively through implicit nodes (those not in the
    # reverse map) to reach the user-facing ids they feed. ``_inputs`` itself
    # is deliberately not read directly: for DataRetrievalOp it equals the
    # declared inputs, which would mask declared-but-unused inputs.
    def collect_config_refs(node: dict[str, Any]) -> set[str]:
        refs: set[str] = set()
        for key, value in node.items():
            if key.startswith("_"):
                continue
            if isinstance(value, str):
                refs.add(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        refs.add(item)
                    elif isinstance(item, dict):
                        refs.update(collect_config_refs(item))
            elif isinstance(value, dict):
                refs.update(collect_config_refs(value))
        return refs

    def resolve_to_user(internal: str) -> set[str]:
        user = internal_to_user.get(internal)
        if user is not None:
            return {user}
        # Implicit node (MessageOp/FormatOp): follow its deps through the graph.
        implicit = native.get(internal)
        if implicit is None:
            return set()
        resolved: set[str] = set()
        for dep in implicit.get("_inputs", []):
            if isinstance(dep, str):
                resolved.update(resolve_to_user(dep))
        return resolved

    config_refs = collect_config_refs(native_op)
    # Op types with no config-based node references (DataOp/LambdaOp) use
    # ``inputs`` purely for closure/leaf computation; there is no dataflow
    # contract to enforce. Derived from the parsed graph, not a name list.
    if not any(internal in native for internal in config_refs):
        return
    native_deps: set[str] = set()
    for internal in config_refs:
        native_deps.update(resolve_to_user(internal))
    native_deps.discard(INPUT_NODE_ID)

    for ref in declared:
        if ref not in native_deps:
            raise DriverProtocolError(
                f"subgraph op {op_id!r} declares input {ref!r} but its "
                f"{op_type} config does not reference it"
            )
    for ref in native_deps:
        if ref not in declared:
            raise DriverProtocolError(
                f"subgraph op {op_id!r} config references {ref!r} but it is "
                f"missing from the declared inputs"
            )


def validate_emitted_subgraph(
    subgraph: list[dict[str, Any]],
    node_registry: dict[str, dict[str, Any]],
    max_nodes: int,
    library: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Structurally validate a planner-emitted subgraph.

    Enforces: node count within ``max_nodes``, known op types (or a ``ref`` to
    a known library entry), unique ids that do not collide with the fixed round
    graph, explicit inputs that reference either a prior-round node (in
    ``node_registry``) or another node in this subgraph, and acyclicity. Raises
    :class:`DriverProtocolError` on any violation.

    ``node_registry`` maps prior-round node ids to their RESOLVED op configs,
    so a cross-round reference resolves to the real op (with its real type),
    not a stub.
    """
    if len(subgraph) > max_nodes:
        raise DriverProtocolError(
            f"subgraph exceeds max_nodes_per_round={max_nodes}: got {len(subgraph)}"
        )
    if not subgraph:
        raise DriverProtocolError("subgraph must not be empty")

    # Resolve library refs first so validation (including declared-input checks)
    # runs against the full op config, not the raw ref envelope.
    resolved = resolve_subgraph(subgraph, library)

    ids: set[str] = set()
    types: dict[str, str] = {}
    # Current-round ops plus prior-round registry ops, so a reference to either
    # resolves to its real op with its real type.
    available_ops: dict[str, dict[str, Any]] = {
        op["id"]: op for op in resolved if isinstance(op.get("id"), str)
    }
    available_ops.update(node_registry)
    for index, op in enumerate(resolved):
        if not isinstance(op, dict):
            raise DriverProtocolError(f"subgraph op at index {index} must be a mapping")
        op_id = op.get("id")
        if not isinstance(op_id, str) or not op_id:
            raise DriverProtocolError(
                f"subgraph op at index {index} requires a non-empty 'id'"
            )
        if op_id in FIXED_ROUND_NODE_IDS:
            raise DriverProtocolError(
                f"subgraph op id {op_id!r} is reserved for the fused round graph"
            )
        if op_id in node_registry:
            raise DriverProtocolError(
                f"subgraph op id {op_id!r} collides with an existing node; "
                "nodes are immutable once created"
            )
        if op_id in ids:
            raise DriverProtocolError(f"duplicate subgraph op id {op_id!r}")
        ids.add(op_id)
        op_type = op.get("op")
        if not isinstance(op_type, str) or op_type not in SUPPORTED_OPS:
            raise DriverProtocolError(
                f"subgraph op {op_id!r} has unsupported op type {op_type!r}; "
                f"supported: {sorted(SUPPORTED_OPS)}"
            )
        types[op_id] = op_type
        raw_inputs = op.get("inputs", [])
        if not isinstance(raw_inputs, list) or not all(
            isinstance(ref, str) for ref in raw_inputs
        ):
            raise DriverProtocolError(
                f"subgraph op {op_id!r} inputs must be a list of node ids"
            )
        _validate_declared_inputs(op_id, op_type, op, raw_inputs, available_ops)

    # Acyclicity: a node may only reference nodes that precede it in the
    # emitted order (the planner emits a topologically sorted DAG). References
    # to prior-round nodes (in ``node_registry``) or the input are always
    # valid; references to other subgraph nodes must already have been emitted
    # (be in ``seen``).
    seen: set[str] = set()
    for op in resolved:
        op_id = op["id"]
        for ref in op.get("inputs", []):
            if ref == INPUT_NODE_ID or ref in node_registry:
                continue
            if ref not in ids:
                raise DriverProtocolError(
                    f"subgraph op {op_id!r} references unknown node id {ref!r}"
                )
            if ref not in seen:
                raise DriverProtocolError(
                    f"subgraph op {op_id!r} references {ref!r} which is not "
                    "emitted before it; subgraph must be acyclic"
                )
        seen.add(op_id)

    # Every leaf is wrapped in an OutputOp, and the runtime only permits
    # OutputOp sources that are LLMOp or DataRetrievalOp. Reject subgraphs whose
    # leaves are any other op type so the round fails at validation, not compile.
    consumed: set[str] = set()
    for op in resolved:
        for ref in op.get("inputs", []):
            if isinstance(ref, str):
                consumed.add(ref)
    for op in resolved:
        op_id = op["id"]
        if op_id in consumed:
            continue
        if types[op_id] not in _OUTPUT_SOURCE_OPS:
            raise DriverProtocolError(
                f"subgraph leaf {op_id!r} is a {types[op_id]}, which cannot be "
                "archived; every leaf must be an LLMOp or DataRetrievalOp"
            )


_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_plan_wrappers(value: str) -> str:
    """Strip a reasoning model's wrappers from around the plan JSON.

    A reasoning model emits its chain of thought in a ``<think>`` block and
    often fences the answer; both wrap the plan without changing it.
    """
    stripped = _THINK_BLOCK_RE.sub("", value)
    fenced = _FENCE_RE.match(stripped)
    return fenced.group(1) if fenced else stripped.strip()


def system_message(
    goal: str,
    observations: list[str],
    topology: list[str],
    threshold: float | None = None,
    library: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render the proposer system message: goal + prior observations + topology."""
    lines = [
        "You are the planner for a dynamic analysis workflow. Each round you "
        "emit a small acyclic subgraph of ops that advances the goal, or STOP "
        "when the accumulated evidence is sufficient.",
        "",
        f"GOAL: {goal}",
    ]
    if observations:
        lines.append("")
        lines.append("PRIOR OBSERVATIONS:")
        for index, observation in enumerate(observations):
            lines.append(f"--- observation {index} ---")
            lines.append(observation)
    lines.append("")
    lines.append("AVAILABLE NODES (existing, immutable, reference by id):")
    lines.append(", ".join([INPUT_NODE_ID, *topology]))
    lines.append(
        f"{INPUT_NODE_ID} is the run's input node, carrying the symbol under "
        "analysis; wire it to any op that needs the symbol."
    )
    lines.append(
        'Every "id" you emit must be NEW and must not be any id already '
        "listed in AVAILABLE NODES. To use an existing node, name it in "
        '"inputs" only — never declare it again as an op. For example, if '
        "top_sector_1 already exists, the correct plan is a single op "
        '{"id": "peers_in_sector_1", "ref": "peers_in_sector", '
        '"inputs": ["top_sector_1"]} — not one that also declares '
        "top_sector_1 again."
    )
    lines.append("")
    lines.append(
        'Emit a structured plan: either {"next": "STOP"} or '
        '{"next": "subgraph", "ops": [...]} where each op is '
        '{"id": <unique id>, "op": <op type>, "inputs": [<node ids>], '
        "...fields}. Each op's inputs must reference an existing node id or "
        "another op id in the same subgraph, and the subgraph must be acyclic. "
        "Available op types: " + ", ".join(sorted(SUPPORTED_OPS)) + "."
    )
    if library:
        lines.append("")
        lines.append("REFERENCE LIBRARY (pre-configured op templates):")
        for ref, template in library.items():
            op_type = template.get("op", "?")
            lines.append(f"- {ref}: {op_type}")
            for key, value in template.items():
                if key == "op":
                    continue
                rendered = str(value).replace("\n", " ")
                lines.append(f"    {key}: {rendered[:200]}")
            lines.append(
                f'    To use it, emit {{"id": <unique id>, "ref": "{ref}", '
                '"inputs": [<node ids>]} — key "ref", never "op"; the '
                "template fills in the rest."
            )
            bindings = template.get("data_spec", {}).get("param_bindings")
            if isinstance(bindings, list) and bindings:
                lines.append(
                    "    Its inputs wire its declared param slots in order; "
                    "the template config is otherwise immutable."
                )
    if threshold is not None:
        lines.append("")
        lines.append(
            f"Stop when the observed evidence meets this sufficiency threshold: "
            f"{threshold}."
        )
    return "\n".join(lines)


def compute_observation(leaf_outputs: dict[str, list[str]], preview_width: int) -> str:
    """Compute the round observation string from the archived leaf outputs.

    The observation LambdaOp runs in-graph to feed the proposer, but its output
    is not an archived output (only LLMOp/DataRetrievalOp can be). The server
    recomputes the same compact summary from the leaf outputs (archived via the
    per-leaf OutputOps) so it can be accumulated across rounds.
    """
    source = observation_lambda(preview_width=preview_width)
    namespace: dict[str, Any] = {"json": json}
    exec(source, namespace)  # noqa: S102 - sandbox-safe aggregation template
    observe = namespace["observe"]
    # Mirror the in-graph framing: one entry per leaf, each a list of the
    # archived (JSON-serialized) output values. observe normalizes each entry.
    per_leaf: list[Any] = [leaf_outputs[leaf_id] for leaf_id in sorted(leaf_outputs)]
    return observe(per_leaf)


def result_outputs(result: Any) -> dict[str, Any]:
    """Extract and validate the nested outputs envelope from a job result."""
    if not isinstance(result, dict):
        raise DriverProtocolError(
            f"job result envelope must be a dict, got {type(result).__name__}"
        )
    nested = result.get("result")
    if not isinstance(nested, dict):
        raise DriverProtocolError(
            "job result envelope must contain a dict 'result' field"
        )
    outputs = nested.get("outputs")
    if not isinstance(outputs, dict):
        raise DriverProtocolError(
            "job result 'result' field must contain a dict 'outputs' field"
        )
    return outputs


def validate_plan(raw_outputs: dict[str, Any]) -> StopPlan | SubgraphPlan:
    """Parse and validate the proposer's plan from a job result.

    Requires exactly one ``plan`` output carrying one serialized value, with
    ``next`` drawn from ``{STOP, SUBGRAPH}``. Anything malformed raises
    :class:`DriverProtocolError` rather than degrading to a silent success.
    """
    plan_outputs: list[Any] = []
    for graph_outputs in raw_outputs.values():
        if not isinstance(graph_outputs, dict):
            raise DriverProtocolError(
                "malformed graph outputs: expected dict, got "
                f"{type(graph_outputs).__name__}"
            )
        if "plan" in graph_outputs:
            plan_outputs.append(graph_outputs["plan"])
    if not plan_outputs:
        raise DriverProtocolError("job result has no 'plan' output")
    if len(plan_outputs) != 1:
        raise DriverProtocolError(
            f"expected exactly one 'plan' output across the result, "
            f"got {len(plan_outputs)}"
        )
    plan_values = plan_outputs[0]
    if not isinstance(plan_values, list):
        raise DriverProtocolError(
            f"'plan' output must be a list, got {type(plan_values).__name__}"
        )
    if len(plan_values) != 1:
        raise DriverProtocolError(
            f"expected exactly one plan value, got {len(plan_values)}"
        )
    value = plan_values[0]
    if not isinstance(value, str):
        raise DriverProtocolError(
            f"plan value must be a serialized string, got {type(value).__name__}"
        )
    try:
        plan = json.loads(_strip_plan_wrappers(value))
    except (ValueError, TypeError) as exc:
        raise DriverProtocolError(f"plan is not valid JSON: {exc}") from exc
    if not isinstance(plan, dict):
        raise DriverProtocolError(
            f"plan must decode to an object, got {type(plan).__name__}"
        )
    try:
        raw = _RawPlan.model_validate(plan)
    except Exception as exc:
        raise DriverProtocolError(f"malformed plan: {exc}") from exc

    if raw.next == STOP:
        if raw.ops:
            raise DriverProtocolError("STOP plan must not include ops")
        return StopPlan()
    if raw.next == SUBGRAPH:
        if raw.ops is None:
            raise DriverProtocolError("subgraph plan requires an 'ops' list")
        return SubgraphPlan(ops=raw.ops)
    raise DriverProtocolError(
        f"plan 'next' must be one of {sorted([STOP, SUBGRAPH])}, got {raw.next!r}"
    )


def plan_to_dict(plan: StopPlan | SubgraphPlan) -> dict[str, Any]:
    """Render a validated plan back to a plain dict."""
    if isinstance(plan, StopPlan):
        return {"next": STOP}
    return {"next": SUBGRAPH, "ops": plan.ops}


def _closure_ops(
    subgraph: list[dict[str, Any]], node_registry: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the ops needed for one round.

    Includes the current subgraph ops plus the transitive closure of the
    prior-round nodes they reference (from ``node_registry``). The YAML parser
    repairs ordering, so the emitted order here need not be topological.
    """
    needed: dict[str, dict[str, Any]] = {}

    def visit(op: dict[str, Any]) -> None:
        for ref in op.get("inputs", []):
            if isinstance(ref, str) and ref in node_registry and ref not in needed:
                needed[ref] = node_registry[ref]
                visit(node_registry[ref])

    for op in subgraph:
        needed[op["id"]] = op
        visit(op)
    return list(needed.values())


def _current_leaves(subgraph: list[dict[str, Any]]) -> list[str]:
    """Return the ids of current-subgraph nodes no other current node consumes."""
    consumed: set[str] = set()
    for op in subgraph:
        for ref in op.get("inputs", []):
            if isinstance(ref, str):
                consumed.add(ref)
    return [op["id"] for op in subgraph if op["id"] not in consumed]


def build_round(
    subgraph: list[dict[str, Any]],
    *,
    node_registry: dict[str, dict[str, Any]],
    round_index: int,
    goal: str,
    observations: list[str],
    topology: list[str],
    preview_width: int,
    model: str,
    max_tokens: int,
    temperature: float,
    threshold: float | None = None,
    library: dict[str, dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> RoundBuild:
    """Assemble the native graph for one round.

    The round graph is the emitted subgraph (converted to native nodes via the
    YAML parser) plus one OutputOp per leaf, the observation LambdaOp, and the
    proposer LLMChatOp. The proposer sees the goal, all prior observations, and
    the topology. Returns the graph plus the ordered archived leaf output names.
    """
    subgraph = resolve_subgraph(subgraph, library)
    closure = _closure_ops(subgraph, node_registry)
    workflow = {
        "name": f"round_{round_index}",
        "inputs": {INPUT_NODE_ID: []},
        "ops": closure,
        "outputs": [],
    }
    parsed = parse_yaml_payload(workflow)
    graph_name = next(iter(parsed))
    native = parsed[graph_name]["graph"]
    # Sort leaf ids so the in-graph observation (blocks.py) and the server-side
    # compute_observation agree on a single deterministic order.
    leaf_ids = sorted(
        _internal_id(graph_name, op["op"], op["id"])
        for op in subgraph
        if op["id"] in _current_leaves(subgraph)
    )
    system = system_message(goal, observations, topology, threshold, library)
    user = (
        "Emit the first subgraph that starts to advance the goal."
        if not subgraph
        else "Here is the observation from the last round. Based on it, emit "
        "the next subgraph, or STOP."
    )
    graph = fused_round_graph(
        native,
        leaf_ids=leaf_ids,
        proposer_system=system,
        proposer_user=user,
        lambda_code=observation_lambda(preview_width=preview_width),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        chat_template_kwargs=chat_template_kwargs,
    )
    return RoundBuild(
        graph=graph,
        leaf_output_names=[f"leaf_{leaf_id}" for leaf_id in leaf_ids],
    )


def _internal_id(scope: str, op_type: str, user_id: str) -> str:
    """Derive the native op id the YAML parser assigns for ``(scope, op, id)``."""
    return make_id(scope, _op_id_prefix(op_type), user_id)
