"""Tests for the dynamic-workflow driver: subgraph validation, plan parsing,
and round graph assembly."""

import json
from typing import Any

import pytest

from lumilake_server.dynamic.blocks import INPUT_NODE_ID
from lumilake_server.dynamic.driver import (
    STOP,
    SUBGRAPH,
    DriverProtocolError,
    StopPlan,
    SubgraphPlan,
    build_round,
    compute_observation,
    plan_to_dict,
    resolve_subgraph,
    result_outputs,
    system_message,
    validate_emitted_subgraph,
    validate_library,
    validate_plan,
)

_LIBRARY: dict[str, dict[str, Any]] = {
    "sector_market_cap": {
        "op": "DataRetrievalOp",
        "data_spec": {
            "type": "lumid",
            "mode": "sql",
            "output_format": "jsonl",
            "verify": False,
            "template": "SELECT sector, AVG(market_cap) FROM reference.profile "
            "GROUP BY sector",
        },
    },
    "top_peers": {
        "op": "DataRetrievalOp",
        "data_spec": {
            "type": "lumid",
            "mode": "sql",
            "output_format": "jsonl",
            "verify": False,
            "template": "SELECT symbol, peer_symbol FROM reference.peers "
            "WHERE symbol = {symbol}",
            "params": [{"label": "symbol", "node": "Symbols"}],
            "param_bindings": [{"label": "symbol", "input": 0}],
        },
    },
    "greeting": {
        "op": "MessageOp",
        "messages": [{"role": "user", "content": "hello"}],
    },
}

_RETRIEVAL = {
    "id": "q1",
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


def _plan_output(plan: dict) -> dict[str, dict[str, list[str]]]:
    return {"round": {"plan": [json.dumps(plan)]}}


# --- validate_emitted_subgraph ---


def test_validate_subgraph_accepts_valid() -> None:
    validate_emitted_subgraph([_RETRIEVAL], {}, 8)


def test_validate_subgraph_rejects_too_many_nodes() -> None:
    ops = [
        {"id": f"n{i}", "op": "DataOp", "inputs": [], "data": ["x"]} for i in range(3)
    ]
    with pytest.raises(DriverProtocolError, match="max_nodes_per_round"):
        validate_emitted_subgraph(ops, {}, 2)


def test_validate_subgraph_rejects_unknown_op() -> None:
    with pytest.raises(DriverProtocolError, match="unsupported op type"):
        validate_emitted_subgraph([{"id": "x", "op": "Nope", "inputs": []}], {}, 8)


def test_validate_subgraph_rejects_unknown_input() -> None:
    with pytest.raises(DriverProtocolError, match="unknown node id"):
        validate_emitted_subgraph(
            [{"id": "x", "op": "DataOp", "inputs": ["missing"], "data": ["a"]}],
            {},
            8,
        )


def test_validate_subgraph_accepts_prior_round_input() -> None:
    validate_emitted_subgraph([_RETRIEVAL], {"q0": _RETRIEVAL}, 8)


def test_validate_subgraph_rejects_cycle() -> None:
    ops = [
        {"id": "a", "op": "DataOp", "inputs": ["b"], "data": ["x"]},
        {"id": "b", "op": "DataOp", "inputs": ["a"], "data": ["x"]},
    ]
    with pytest.raises(DriverProtocolError, match="acyclic"):
        validate_emitted_subgraph(ops, {}, 8)


def _retrieval(id: str, inputs: list[str]) -> dict:
    """A DataRetrievalOp whose params reference its declared inputs."""
    return {
        "id": id,
        "op": "DataRetrievalOp",
        "inputs": inputs,
        "data_spec": {
            "params": [
                {"label": "s", "node": ref} for ref in inputs if ref != "Symbols"
            ]
        },
    }


def test_validate_subgraph_accepts_valid_forward_ref() -> None:
    # b consumes a; a precedes b, so this is a valid topological order.
    ops = [
        _retrieval("a", ["Symbols"]),
        _retrieval("b", ["a"]),
    ]
    validate_emitted_subgraph(ops, {}, 8)


def test_validate_subgraph_rejects_backward_ref() -> None:
    # a consumes b, but b is emitted after a — a forward reference, invalid.
    ops = [
        _retrieval("a", ["b"]),
        _retrieval("b", ["Symbols"]),
    ]
    with pytest.raises(DriverProtocolError, match="acyclic"):
        validate_emitted_subgraph(ops, {}, 8)


def test_validate_subgraph_rejects_non_output_leaf() -> None:
    # A DataOp leaf is wrapped in an OutputOp, which the runtime rejects.
    ops = [{"id": "a", "op": "DataOp", "inputs": [], "data": ["x"]}]
    with pytest.raises(DriverProtocolError, match="cannot be archived"):
        validate_emitted_subgraph(ops, {}, 8)


def test_validate_subgraph_accepts_output_leaf() -> None:
    ops = [
        _retrieval("a", ["Symbols"]),
        {
            "id": "b",
            "op": "LLMChatOp",
            "inputs": ["a"],
            "config": {"model": "Qwen/Qwen3-8B"},
            "messages": [{"role": "user", "content": "a"}],
        },
    ]
    validate_emitted_subgraph(ops, {}, 8)


def test_validate_subgraph_accepts_vision_leaf() -> None:
    # LLMVisionOp is an LLMOp subclass, so the runtime accepts it as an
    # OutputOp source; validation must not reject it.
    ops = [
        _retrieval("a", ["Symbols"]),
        {
            "id": "b",
            "op": "LLMVisionOp",
            "inputs": ["a"],
            "config": {"model": "Qwen/Qwen3-8B"},
            "image_source": "a",
            "messages": [{"role": "user", "content": "a"}],
        },
    ]
    validate_emitted_subgraph(ops, {}, 8)


def test_validate_subgraph_accepts_image_gen_leaf() -> None:
    # ImageGenerationOp is an LLMOp subclass, so it is a valid OutputOp source.
    ops = [
        _retrieval("a", ["Symbols"]),
        {
            "id": "b",
            "op": "ImageGenerationOp",
            "inputs": ["a"],
            "config": {"model": "Qwen/Qwen3-8B"},
            "content": "a",
        },
    ]
    validate_emitted_subgraph(ops, {}, 8)


def test_validate_subgraph_rejects_unconsumed_input() -> None:
    # An LLMChatOp whose declared input is not referenced by its messages must
    # be rejected: the declared wiring must agree with native dependencies.
    ops = [
        _retrieval("a", ["Symbols"]),
        {
            "id": "b",
            "op": "LLMChatOp",
            "inputs": ["a"],
            "config": {"model": "Qwen/Qwen3-8B"},
            "messages": [{"role": "user", "content": "unrelated"}],
        },
    ]
    with pytest.raises(DriverProtocolError):
        validate_emitted_subgraph(ops, {}, 8)


def test_validate_subgraph_rejects_native_but_undeclared() -> None:
    # A retrieval op whose params reference a node NOT in its declared inputs
    # must be rejected: the native dependency set must equal the declared set.
    ops = [
        _retrieval("a", ["Symbols"]),
        {
            "id": "b",
            "op": "DataRetrievalOp",
            "inputs": ["a"],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "output_format": "jsonl",
                "verify": False,
                "template": "SELECT * FROM t WHERE x = {p1} AND y = {p2}",
                "params": [
                    {"label": "p1", "node": "a"},
                    {"label": "p2", "node": "c"},
                ],
            },
        },
        _retrieval("c", ["Symbols"]),
    ]
    with pytest.raises(DriverProtocolError, match="not in its declared inputs"):
        validate_emitted_subgraph(ops, {}, 8)


def test_validate_subgraph_rejects_declared_but_unused() -> None:
    # A retrieval op declaring an input its params do not reference must be
    # rejected.
    ops = [
        _retrieval("a", ["Symbols"]),
        {
            "id": "b",
            "op": "DataRetrievalOp",
            "inputs": ["a", "c"],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "output_format": "jsonl",
                "verify": False,
                "template": "SELECT * FROM t WHERE x = {p1}",
                "params": [{"label": "p1", "node": "a"}],
            },
        },
        _retrieval("c", ["Symbols"]),
    ]
    with pytest.raises(DriverProtocolError, match="does not reference"):
        validate_emitted_subgraph(ops, {}, 8)


def test_validate_subgraph_rejects_prior_round_id() -> None:
    # Re-emitting a prior-round node id violates the immutable-node contract.
    ops = [
        {"id": "q1", "op": "DataRetrievalOp", "inputs": ["Symbols"], "data_spec": {}}
    ]
    with pytest.raises(DriverProtocolError, match="immutable"):
        validate_emitted_subgraph(ops, {"q1": _RETRIEVAL}, 8)


def test_validate_subgraph_rejects_duplicate_ids() -> None:
    ops = [
        {"id": "a", "op": "DataOp", "inputs": [], "data": ["x"]},
        {"id": "a", "op": "DataOp", "inputs": [], "data": ["y"]},
    ]
    with pytest.raises(DriverProtocolError, match="duplicate"):
        validate_emitted_subgraph(ops, {}, 8)


def test_validate_subgraph_rejects_reserved_id() -> None:
    with pytest.raises(DriverProtocolError, match="reserved"):
        validate_emitted_subgraph(
            [{"id": "observation", "op": "DataOp", "inputs": [], "data": ["x"]}],
            {},
            8,
        )


def test_validate_subgraph_rejects_empty() -> None:
    with pytest.raises(DriverProtocolError, match="must not be empty"):
        validate_emitted_subgraph([], {}, 8)


# --- validate_plan ---


def test_validate_plan_stop() -> None:
    plan = validate_plan(_plan_output({"next": STOP}))
    assert isinstance(plan, StopPlan)


def test_validate_plan_subgraph() -> None:
    plan = validate_plan(_plan_output({"next": SUBGRAPH, "ops": [_RETRIEVAL]}))
    assert isinstance(plan, SubgraphPlan)
    assert plan.ops == [_RETRIEVAL]


def test_validate_plan_rejects_stop_with_ops() -> None:
    with pytest.raises(DriverProtocolError, match="STOP"):
        validate_plan(_plan_output({"next": STOP, "ops": [_RETRIEVAL]}))


def test_validate_plan_rejects_unknown_next() -> None:
    with pytest.raises(DriverProtocolError, match="'next'"):
        validate_plan(_plan_output({"next": "bogus"}))


def test_validate_plan_rejects_missing_ops() -> None:
    with pytest.raises(DriverProtocolError, match="ops"):
        validate_plan(_plan_output({"next": SUBGRAPH}))


def test_validate_plan_rejects_malformed_json() -> None:
    with pytest.raises(DriverProtocolError, match="not valid JSON"):
        validate_plan({"round": {"plan": ["not json"]}})


def test_validate_plan_rejects_no_plan_output() -> None:
    with pytest.raises(DriverProtocolError, match="no 'plan' output"):
        validate_plan({"round": {}})


def test_plan_to_dict_roundtrip() -> None:
    assert plan_to_dict(StopPlan()) == {"next": STOP}
    assert plan_to_dict(SubgraphPlan(ops=[_RETRIEVAL])) == {
        "next": SUBGRAPH,
        "ops": [_RETRIEVAL],
    }


# --- result_outputs ---


def test_result_outputs_extracts_envelope() -> None:
    outputs = result_outputs({"result": {"outputs": {"round": {"plan": ["x"]}}}})
    assert outputs == {"round": {"plan": ["x"]}}


def test_result_outputs_rejects_malformed() -> None:
    with pytest.raises(DriverProtocolError):
        result_outputs({"result": {}})


# --- system_message ---


def test_system_message_includes_goal_observations_topology() -> None:
    msg = system_message("my goal", ["obs1", "obs2"], ["n1", "n2"])
    assert "my goal" in msg
    assert "obs1" in msg
    assert "obs2" in msg
    assert "n1, n2" in msg
    assert "STOP" in msg


def test_system_message_empty_observations() -> None:
    msg = system_message("goal", [], [])
    assert "PRIOR OBSERVATIONS" not in msg
    # With no accumulated nodes the run's input node is still available to
    # wire to, so the planner is told about it rather than "(none yet)".
    assert INPUT_NODE_ID in msg


# --- compute_observation ---


def test_compute_observation() -> None:
    obs = compute_observation({"leaf1": ['{"a": 1}', '{"a": 2}']}, preview_width=900)
    assert obs.startswith("rows=")


def test_compute_observation_produces_numeric_stats() -> None:
    # Leaf outputs are archived as JSON-serialized records coerced to strings;
    # compute_observation must decode them so numeric stats are produced.
    obs = compute_observation(
        {"leaf1": ['{"market_cap": 10}', '{"market_cap": 20}']}, preview_width=900
    )
    assert "market_cap" in obs
    assert "sum=30" in obs
    assert "avg=15" in obs


def test_compute_observation_llm_text_leaf() -> None:
    # A plain-text LLM leaf is not JSON; it must be treated as text, not fail.
    obs = compute_observation({"leaf1": ["hello world"]}, preview_width=900)
    assert "rows=1" in obs
    assert "hello world" in obs


def test_compute_observation_malformed_json_as_text() -> None:
    # Malformed JSON must be treated as text, not raise.
    obs = compute_observation({"leaf1": ["not json {"]}, preview_width=900)
    assert "rows=1" in obs


def test_compute_observation_two_leaves_deterministic() -> None:
    # Two leaves, each one value; order is deterministic by leaf id.
    obs = compute_observation(
        {
            "leaf_b": ['{"x": 5}'],
            "leaf_a": ['{"x": 1}'],
        },
        preview_width=900,
    )
    assert "rows=2" in obs
    assert "sum=6" in obs


def test_compute_observation_unwraps_df_envelope() -> None:
    # A retrieval leaf arrives as {"df": "<json string>"} whose value is a
    # column-oriented table. The envelope must be unwrapped and transposed so
    # the observation reports the real row count and numeric stats, not one
    # row per leaf holding the raw JSON string.
    obs = compute_observation(
        {
            "leaf1": [
                '{"df": "{\\"avg_market_cap\\": {\\"0\\": 44683131633.8, '
                '\\"1\\": 28540212749.0}, \\"sector\\": {\\"0\\": '
                '\\"Communication Services\\", \\"1\\": \\"Technology\\"}}"}'
            ]
        },
        preview_width=900,
    )
    assert "rows=2" in obs
    assert "avg_market_cap" in obs
    assert "sum=7.322e+10" in obs
    assert "avg=3.661e+10" in obs


def test_compute_observation_two_tables_do_not_mix() -> None:
    # Two unrelated retrieval tables, each transposed on its own. The row count
    # is the sum across leaves, no preview record carries columns from more
    # than one table, and both leaves' numeric stats are still reported.
    sectors = (
        '{"df": "{\\"sector\\": {\\"0\\": \\"Communication Services\\", '
        '\\"1\\": \\"Technology\\"}, \\"avg_market_cap\\": {\\"0\\": '
        '44683131633.8, \\"1\\": 28540212749.0}}"}'
    )
    peers = (
        '{"df": "{\\"symbol\\": {\\"0\\": \\"NVDA\\", \\"1\\": \\"AVGO\\"}, '
        '\\"market_cap\\": {\\"0\\": 1826303491551.5, \\"1\\": '
        '1040000000000.0}}"}'
    )
    obs = compute_observation(
        {"leaf_a": [sectors], "leaf_b": [peers]}, preview_width=400
    )
    assert "rows=4" in obs
    assert "avg_market_cap" in obs
    assert "market_cap" in obs
    preview = json.loads(obs.split("preview=", 1)[1])
    # Each preview record comes from exactly one table; no record mixes
    # sector/avg_market_cap with symbol/market_cap.
    for record in preview:
        assert not ({"sector", "avg_market_cap"} & set(record)) or not (
            {"symbol", "market_cap"} & set(record)
        )


# --- build_round ---


def test_build_round_round0_no_subgraph() -> None:
    round_build = build_round(
        [],
        node_registry={},
        round_index=0,
        goal="g",
        observations=[],
        topology=[],
        preview_width=900,
        model="Qwen/Qwen3-8B",
        max_tokens=768,
        temperature=0.4,
    )
    graph = round_build.graph
    assert "proposer" in graph
    assert "observation" not in graph
    assert round_build.leaf_output_names == []


def test_build_round_with_subgraph() -> None:
    round_build = build_round(
        [_RETRIEVAL],
        node_registry={},
        round_index=1,
        goal="g",
        observations=["obs1"],
        topology=[],
        preview_width=900,
        model="Qwen/Qwen3-8B",
        max_tokens=768,
        temperature=0.4,
    )
    graph = round_build.graph
    assert "observation" in graph
    assert "proposer" in graph
    # The leaf gets an OutputOp.
    leaf_outputs = [k for k in graph if k.startswith("output_")]
    assert leaf_outputs
    assert len(round_build.leaf_output_names) == 1
    assert round_build.leaf_output_names[0].startswith("leaf_")


def test_build_round_includes_prior_round_nodes() -> None:
    prior = {"q1": _RETRIEVAL}
    subgraph = [
        {
            "id": "q2",
            "op": "DataRetrievalOp",
            "inputs": ["q1"],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "output_format": "jsonl",
                "verify": False,
                "template": "SELECT * FROM y",
                "params": [{"label": "s", "node": "q1"}],
            },
        }
    ]
    round_build = build_round(
        subgraph,
        node_registry=prior,
        round_index=2,
        goal="g",
        observations=[],
        topology=["q1"],
        preview_width=900,
        model="Qwen/Qwen3-8B",
        max_tokens=768,
        temperature=0.4,
    )
    graph = round_build.graph
    # Both the prior q1 and current q2 are present.
    assert any("q1" in k for k in graph)
    assert any("q2" in k for k in graph)


# --- reference library ---


def test_validate_library_accepts_valid() -> None:
    validate_library(_LIBRARY)


def test_validate_library_accepts_none() -> None:
    validate_library(None)
    validate_library({})


def test_validate_library_rejects_unknown_op() -> None:
    with pytest.raises(DriverProtocolError, match="unsupported op type"):
        validate_library({"bad": {"op": "NotAnOp"}})


def test_validate_library_rejects_non_mapping() -> None:
    with pytest.raises(DriverProtocolError, match="must be a mapping"):
        # Intentionally invalid: a bare string where a mapping is required.
        validate_library({"bad": "not a mapping"})  # type: ignore[dict-item]


def test_validate_subgraph_accepts_library_ref() -> None:
    subgraph = [{"id": "q1", "ref": "sector_market_cap", "inputs": ["Symbols"]}]
    validate_emitted_subgraph(subgraph, {}, 8, _LIBRARY)


def test_validate_subgraph_accepts_ref_with_op() -> None:
    subgraph = [
        {
            "id": "q1",
            "op": "DataRetrievalOp",
            "ref": "sector_market_cap",
            "inputs": ["Symbols"],
        }
    ]
    validate_emitted_subgraph(subgraph, {}, 8, _LIBRARY)


def test_validate_subgraph_rejects_unknown_library_ref() -> None:
    subgraph = [{"id": "q1", "ref": "nope", "inputs": ["Symbols"]}]
    with pytest.raises(DriverProtocolError, match="unknown library entry"):
        validate_emitted_subgraph(subgraph, {}, 8, _LIBRARY)


def test_validate_subgraph_rejects_ref_without_library() -> None:
    subgraph = [{"id": "q1", "ref": "sector_market_cap", "inputs": ["Symbols"]}]
    with pytest.raises(DriverProtocolError, match="unknown library entry"):
        validate_emitted_subgraph(subgraph, {}, 8)


def test_validate_subgraph_rejects_ref_op_mismatch() -> None:
    subgraph = [
        {
            "id": "q1",
            "op": "MessageOp",
            "ref": "sector_market_cap",
            "inputs": ["Symbols"],
        }
    ]
    with pytest.raises(DriverProtocolError, match="declares op"):
        validate_emitted_subgraph(subgraph, {}, 8, _LIBRARY)


def test_resolve_subgraph_expands_ref() -> None:
    subgraph = [{"id": "q1", "ref": "sector_market_cap", "inputs": ["Symbols"]}]
    resolved = resolve_subgraph(subgraph, _LIBRARY)
    assert resolved[0]["op"] == "DataRetrievalOp"
    assert resolved[0]["id"] == "q1"
    assert resolved[0]["inputs"] == ["Symbols"]
    assert "template" in resolved[0]["data_spec"]


def test_resolve_subgraph_passthrough_without_ref() -> None:
    resolved = resolve_subgraph([_RETRIEVAL], _LIBRARY)
    assert resolved == [_RETRIEVAL]


def test_resolve_subgraph_none_library() -> None:
    assert resolve_subgraph([_RETRIEVAL], None) == [_RETRIEVAL]


def test_resolve_subgraph_rejects_unknown_ref() -> None:
    subgraph = [{"id": "q1", "ref": "nope", "inputs": ["Symbols"]}]
    with pytest.raises(DriverProtocolError, match="unknown library entry"):
        resolve_subgraph(subgraph, _LIBRARY)


def test_resolve_subgraph_rejects_config_override() -> None:
    # The planner may not override a library template's config (data_spec).
    subgraph = [
        {
            "id": "q1",
            "ref": "sector_market_cap",
            "inputs": ["Symbols"],
            "data_spec": {"template": "SELECT 1"},
        }
    ]
    with pytest.raises(DriverProtocolError, match="may not override"):
        resolve_subgraph(subgraph, _LIBRARY)


def test_resolve_subgraph_accepts_id_and_inputs_override() -> None:
    subgraph = [{"id": "q1", "ref": "sector_market_cap", "inputs": ["Symbols"]}]
    resolved = resolve_subgraph(subgraph, _LIBRARY)
    assert resolved[0]["id"] == "q1"
    assert resolved[0]["inputs"] == ["Symbols"]
    # The template's config is preserved verbatim.
    assert resolved[0]["data_spec"]["template"].startswith("SELECT sector")


def test_resolve_subgraph_accepts_matching_op_override() -> None:
    # An explicit "op" matching the template is allowed (harmless, explicit).
    subgraph = [
        {
            "id": "q1",
            "op": "DataRetrievalOp",
            "ref": "sector_market_cap",
            "inputs": ["Symbols"],
        }
    ]
    resolved = resolve_subgraph(subgraph, _LIBRARY)
    assert resolved[0]["op"] == "DataRetrievalOp"


def test_resolve_subgraph_wires_param_slot() -> None:
    # The planner wires a retrieval template's {placeholder} to a node purely
    # through inputs; the declared param_binding substitutes it.
    subgraph = [
        {
            "id": "q1",
            "ref": "top_peers",
            "inputs": ["q0"],
        }
    ]
    resolved = resolve_subgraph(subgraph, _LIBRARY)
    params = resolved[0]["data_spec"]["params"]
    assert params == [{"label": "symbol", "node": "q0"}]


def test_resolve_subgraph_preserves_param_path() -> None:
    # peers_in_sector declares a sector param with a drill path; wiring it to a
    # prior node must preserve both the node pointer and the path so the worker
    # resolves <node>.items.table.sector.
    library = {
        "peers_in_sector": {
            "op": "DataRetrievalOp",
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "output_format": "jsonl",
                "verify": False,
                "template": "SELECT * FROM reference.profile "
                "WHERE sector = '{sector}'",
                "params": [
                    {
                        "label": "sector",
                        "node": "top_sector",
                        "path": "items.table.sector",
                    }
                ],
                "param_bindings": [{"label": "sector", "input": 0}],
            },
        }
    }
    subgraph = [{"id": "q1", "ref": "peers_in_sector", "inputs": ["some_prior_node"]}]
    resolved = resolve_subgraph(subgraph, library)
    params = resolved[0]["data_spec"]["params"]
    assert params == [
        {"label": "sector", "node": "some_prior_node", "path": "items.table.sector"}
    ]


def test_resolve_subgraph_rejects_wrong_input_count() -> None:
    # top_peers declares one binding; two inputs must be rejected.
    subgraph = [
        {
            "id": "q1",
            "ref": "top_peers",
            "inputs": ["q0", "q1"],
        }
    ]
    with pytest.raises(DriverProtocolError, match="binding"):
        resolve_subgraph(subgraph, _LIBRARY)


def test_resolve_subgraph_accepts_non_retrieval_ref() -> None:
    # A non-retrieval template (no data_spec) resolves as-is; there are no
    # param slots to wire.
    subgraph = [
        {
            "id": "q1",
            "ref": "greeting",
            "inputs": [],
        }
    ]
    resolved = resolve_subgraph(subgraph, _LIBRARY)
    assert resolved[0]["op"] == "MessageOp"
    assert resolved[0]["id"] == "q1"


def test_system_message_describes_library() -> None:
    msg = system_message("goal", [], [], library=_LIBRARY)
    assert "REFERENCE LIBRARY" in msg
    assert "sector_market_cap" in msg
    assert "top_peers" in msg
    assert "DataRetrievalOp" in msg


def test_system_message_no_duplicate_nodes_instruction() -> None:
    msg = system_message("goal", [], [], library=_LIBRARY)
    assert "AVAILABLE NODES" in msg
    # New ids only: an emitted id must not collide with an existing node.
    assert 'Every "id" you emit must be NEW' in msg
    assert "must not be any id already listed in AVAILABLE NODES" in msg
    # Reuse via inputs only: never declare an existing node again as an op.
    assert 'name it in "inputs" only' in msg
    assert "never declare it again as an op" in msg
    # The concrete example shows the correct single-op shape.
    assert "peers_in_sector_1" in msg
    assert "top_sector_1" in msg


def test_build_round_resolves_library_ref() -> None:
    subgraph = [{"id": "q1", "ref": "sector_market_cap", "inputs": ["Symbols"]}]
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
        library=_LIBRARY,
    )
    graph = round_build.graph
    assert "observation" in graph
    assert "proposer" in graph
    assert any("q1" in k for k in graph)
