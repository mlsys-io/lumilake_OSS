"""Internal graph-assembly machinery for dynamic workflow execution.

A dynamic workflow is a YAML document; there is no user-facing Python class
hierarchy to subclass. Each round the planner emits a subgraph of ops in the
Lumilake YAML convention; this module validates the per-op contract and
assembles the round's native graph (the emitted subgraph plus an observation
LambdaOp and a proposer LLMChatOp). It is an internal implementation detail,
not a public extension point.
"""

from typing import Any

from lumilake_server.common import GenerationConfig
from lumilake_server.ops import (
    DataOp,
    FormatOp,
    LambdaOp,
    LLMChatOp,
    MessageOp,
    OpMessage,
    OutputOp,
)

INPUT_NODE_ID = "Symbols"
OBSERVATION_NODE_ID = "observation"
FORMAT_NODE_ID = "format"
MESSAGE_NODE_ID = "message"
PROPOSER_NODE_ID = "proposer"
OUTPUT_NODE_ID = "output"


def _escape_braces(text: str) -> str:
    """Escape braces in a literal message string.

    The runtime renders every literal message through ``str.format_map``, so a
    bare ``{`` in the plan schema or in a library entry's SQL template is read
    as a replacement field and rejected.
    """
    return text.replace("{", "{{").replace("}", "}}")


def observe(args):
    """Observation LambdaOp entry point; the real body is the ``lambda_code``
    the runtime embeds, so this stub only supplies the function name."""
    raise NotImplementedError("observation body is injected at render time")


def fused_round_graph(
    subgraph: dict[str, dict[str, Any]],
    *,
    leaf_ids: list[str],
    proposer_system: str,
    proposer_user: str,
    lambda_code: str,
    model: str = "Qwen/Qwen3-8B",
    max_tokens: int = 768,
    temperature: float = 0.4,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the native graph for one fused round.

    Layout: ``[subgraph] -> [observation LambdaOp] -> [proposer LLMChatOp]``.
    ``subgraph`` is an ordered mapping of native op dicts (``_id``/``_op``/
    ``_inputs`` form) already validated and topologically sorted. ``leaf_ids``
    names the current round's leaf nodes (a subset of ``subgraph``) to observe
    and archive. Each leaf gets an ``OutputOp`` so its result is archived; the
    leaves are fed directly to the observation LambdaOp (one input per leaf);
    the proposer is a leaf consuming the observation as its sole user message,
    so it sees only the compact stats + preview.
    """
    for leaf_id in leaf_ids:
        if leaf_id not in subgraph:
            raise ValueError(f"leaf id {leaf_id!r} not present in the subgraph")

    ordered: dict[str, Any] = {}
    for node_id, node in subgraph.items():
        ordered[node_id] = node

    # One OutputOp per leaf so each leaf result is archived and retrievable.
    # The path is left unset so the runtime applies its per-op-type default
    # (items.table for sql/agent retrieval, items.content for s3, items.output
    # for LLM leaves) rather than a fixed path that mismatches the leaf shape.
    leaf_outputs: dict[str, dict[str, Any]] = {}
    for leaf_id in leaf_ids:
        leaf_ref = DataOp(data=[])
        leaf_ref.id = leaf_id
        output = OutputOp(name=f"leaf_{leaf_id}", output=leaf_ref)
        output.id = f"output_{leaf_id}"
        leaf_outputs[output.id] = output.serialize()
    for node_id, node in leaf_outputs.items():
        ordered[node_id] = node

    # Feed each leaf directly to the observation LambdaOp. The leaves are
    # heterogeneous (retrieval records vs. LLM text), so observe receives a
    # list of per-leaf outputs and normalizes each; a newline-joined string
    # would not be parseable JSON.
    if leaf_ids:
        leaf_refs = [DataOp(data=[]) for _ in leaf_ids]
        for ref, leaf_id in zip(leaf_refs, leaf_ids):
            ref.id = leaf_id

        # A retrieval leaf reaches the sandbox as a table, which it rejects:
        # a lambda argument must be text. FormatOp is where the runtime turns a
        # table into text, so each leaf is rendered before it is observed.
        rendered_leaves = []
        for ref, leaf_id in zip(leaf_refs, leaf_ids):
            render_op = FormatOp("{ref0}", ref0=ref)
            render_op.id = f"render_{leaf_id}"
            ordered[render_op.id] = render_op.serialize()
            rendered_leaves.append(render_op)

        observation = LambdaOp(rendered_leaves, observe, code=lambda_code)
        observation.id = OBSERVATION_NODE_ID
        ordered[OBSERVATION_NODE_ID] = observation.serialize()

        format_op = FormatOp(f"{proposer_user}\n\n{{ref0}}", ref0=observation)
        format_op.id = FORMAT_NODE_ID
        ordered[FORMAT_NODE_ID] = format_op.serialize()
        user_content: Any = format_op
    else:
        # Round 0 has no subgraph and no observation; the proposer sees only
        # the system message (goal + topology). The runtime rejects a FormatOp
        # with no inputs, so the user turn is carried as a literal string.
        user_content = _escape_braces(proposer_user)

    message_op = MessageOp(
        [
            OpMessage("system", _escape_braces(proposer_system)),
            OpMessage("user", user_content),
        ]
    )
    message_op.id = MESSAGE_NODE_ID
    ordered[MESSAGE_NODE_ID] = message_op.serialize()

    proposer = LLMChatOp(
        message_op,
        config=GenerationConfig(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            chat_template_kwargs=chat_template_kwargs,
        ),
        return_history=False,
        cacheable=False,
    )
    proposer.id = PROPOSER_NODE_ID
    ordered[PROPOSER_NODE_ID] = proposer.serialize()

    output = OutputOp(name="plan", output=proposer, path="items.output")
    output.id = OUTPUT_NODE_ID
    ordered[OUTPUT_NODE_ID] = output.serialize()
    return ordered
