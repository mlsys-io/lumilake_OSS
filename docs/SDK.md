# SDK Reference

Lumilake ships sync and async Python clients behind the `sdk` extra. The SDK covers common server API surfaces and local deploy helpers. The CLI remains the complete operational command surface.

## Install

```bash
pip install "lumilake[sdk]"
```

Add deploy lifecycle helpers with:

```bash
pip install "lumilake[sdk,deploy]"
```

From a source checkout:

```bash
uv sync --group lint --group test --extra sdk --extra deploy
```

## Clients

```python
from lumilake import LumilakeClient

with LumilakeClient.from_config() as client:
    client.health()
    client.jobs.list()
```

```python
from lumilake import AsyncLumilakeClient

async with AsyncLumilakeClient.from_config() as client:
    await client.health()
    await client.jobs.list()
```

Both clients accept `base_url=` directly or load the server URL with `.from_config()`. See [Configuration](#configuration) for resolution order.

## Configuration

The SDK and CLI share a small TOML file that records the server URL:

```
~/.lumilake/config.toml
```

Schema:

```toml
base_url = "https://lumilake.example.com"
api_key  = "lm_pat_..."  # optional; falls back to LUMILAKE_API_KEY env or empty
```

### What writes it

`lumilake deploy up` writes the file with the local stack URL once the
stack starts. Remote / hosted users should pass `base_url=` explicitly
or set `LUMILAKE_BASE_URL`.

### How clients resolve

`from_config(path=...)` reads a specific config file directly; use it
for tests or non-default installs. The default constructor resolves
`base_url` (and `api_key`) through the precedence below.

### Authentication

Both clients accept `api_key=` and resolve it together with `base_url`
through a single `resolve_config()` call. Precedence:

1. Explicit `base_url=` / `api_key=` constructor args.
2. `LUMILAKE_BASE_URL` / `LUMILAKE_API_KEY` environment variables.
3. Saved `~/.lumilake/config.toml` (written by `lumilake init`).
4. Defaults: `base_url=http://127.0.0.1:9000`, `api_key=None`.

When `api_key` resolves to a value, the SDK sends it as
`Authorization: Bearer <api_key>` on every request. Deployments running
with `LUMILAKE_REQUIRE_IDENTITY_PROVIDER=1` reject requests without it.

```python
from lumilake import LumilakeClient

with LumilakeClient(api_key="lm_pat_live_…") as client:
    client.jobs.list()
```

## Resources

| Surface | Sync | Async |
|---------|------|-------|
| Health | `client.health()` | `await client.health()` |
| Deploy | `client.deploy.<verb>(...)` | `await client.deploy.<verb>(...)` |
| Jobs | `client.jobs.submit / preview / list / list_all / get / progress / result / inputs / artifact / list_workflows / get_logs / stream_logs / download_logs / cancel / wait / watch(...)` | same, await |
| Workers | `client.workers.list / list_all / get(...)` | same, await |
| Traces | `client.traces.list / list_all / get(...)` | same, await |

### Jobs

All `Jobs` / `AsyncJobs` methods mirror the CLI surface and the server's HTTP routes one-to-one:

```python
client.jobs.submit({"data": [...]}, workflow_format="yaml")
# Override the server default optimizer for one job (see GET /api/v1/optimizer):
client.jobs.submit({"data": [...], "optimizer": "halo"}, workflow_format="yaml")
# Override hardware requirements for one job; unset fields fall back to the server's HARDWARE_* env defaults.
client.jobs.submit({"data": [...], "hardware": {"cpu": 32, "memory": "128Gi"}}, workflow_format="yaml")
client.jobs.preview({"data": [...]}, workflow_format="yaml")
# Preview with a specific optimizer — or omit "optimizer" to use the server default:
client.jobs.preview({"data": [...], "optimizer": "topological-sort"}, workflow_format="yaml")
# Preview accepts the same "hardware" object as submit; unset fields fall back to HARDWARE_* env defaults.
client.jobs.preview({"data": [...], "hardware": {"gpu": 1, "gpu_memory": "24Gi"}}, workflow_format="yaml")
client.jobs.list(status="completed", limit=20)
client.jobs.get(job_id)
client.jobs.progress(job_id)
client.jobs.result(job_id)
client.jobs.inputs(job_id)
client.jobs.cancel(job_id)
client.jobs.artifact(job_id, path="s3://...", output="result.json")

# Per-workflow FlowMesh logs (mirrors `lumilake job logs show/stream/download`).
workflows = client.jobs.list_workflows(job_id)
page = client.jobs.get_logs(job_id, workflows[0].workflow_id, limit=200)  # LogQueryResponse
for entry in client.jobs.stream_logs(job_id, workflows[0].workflow_id):  # Iterator[LogEntry]
    print(entry.event.message)
paths = client.jobs.download_logs(job_id, workflows[0].workflow_id, Path("./logs"))  # list[Path]

# Block until a terminal state and return the final job record.
client.jobs.wait(job_id, timeout=900.0)

# Yield one snapshot per poll until the job is terminal.
for snapshot in client.jobs.watch(job_id):
    print(snapshot["status"], snapshot["progress"])
```

### Pagination — `list_all`

`Jobs`, `Workers`, and `Traces` all expose `list_all(...)`; the async versions return async iterators. The iterator handles cursor traversal internally; pass `page_size=` to bound the per-request `limit`, and the iterator stops when the server stops returning a `next_cursor`.

```python
for job in client.jobs.list_all(status="completed", page_size=50):
    process(job)
```

```python
async for job in async_client.jobs.list_all():
    await process(job)
```

### Dynamic workflows

A dynamic workflow is a YAML document with a plaintext `goal` and a `driver`
section; `name` is optional. The server owns the planning-agent loop: there is
no Python block class to subclass and no block catalog. Each round the planner
(an LLM) emits a small acyclic subgraph of ops that advances the goal, or
`STOP`.

Submit a dynamic workflow to the server with `workflow_format="yaml"` (the
YAML's root `type: dynamic` field marks it as dynamic):

```python
from lumilake import LumilakeClient

yaml_text = open("examples/templates/yaml/market-data-dynamic.yaml").read()
payload = {
    "data": [
        {
            "name": "demo",
            "workflow": yaml_text,
            "inputs": {"Symbols": ["NVDA"]},
            "output_location": {"type": "s3", "prefix": "dynamic/market/"},
        }
    ]
}

with LumilakeClient(base_url="http://localhost:9000") as client:
    resp = client.jobs.submit(payload, workflow_format="yaml")
    print(resp["job_id"])
```

The payload is the server's `JobSubmitRequest` shape: a top-level `data` list,
each item carrying `workflow` (the YAML text), `inputs`, and `output_location`.
`output_location` is REQUIRED in every envelope item even when the YAML
declares `driver.output_location` — the server only gives the YAML value
precedence after request-model validation. The async client uses the same
payload with `await client.jobs.submit(...)`.

The server renders round 0, runs each round as a child job, validates each
emitted subgraph, and stops on `STOP` or `max_rounds`. The parent job owns the
run's traces and lifecycle. Each round runs one native graph:

- the emitted subgraph runs first (each leaf is archived under a `leaf_<id>`
  output);
- the observation LambdaOp computes numeric statistics and a bounded preview
  of the subgraph's leaf outputs;
- the proposer LLMChatOp sees the goal, all prior observations, and the overall
  topology, and emits one structured plan: either `{"next": "STOP"}` or
  `{"next": "subgraph", "ops": [...]}`.

Within one run, every node can reference any existing node by id from any
previous round (an accumulated global node registry); nodes are immutable once
created. A run must target exactly one non-empty symbol. The parent's result
carries the per-round plans.

The `driver:` YAML settings the server accepts are:

- `model` — proposer model, required.
- `max_tokens` — proposer token limit, default `768`; must be a positive integer.
- `temperature` — proposer sampling temperature, default `0.4`.
- `preview_width` — observation preview character limit, default `900`; must be a positive integer.
- `job_timeout` — per-job wait timeout in seconds, default `600.0`; must be greater than zero.
- `max_rounds` — round limit, default `10`.
- `max_nodes_per_round` — per-round subgraph node limit, default `8`.
- `threshold` — sufficiency threshold, default `None`.
- `chat_template_kwargs` — chat-template kwargs for the proposer, default
  `None`. Use it to turn off a reasoning model's thinking mode (Qwen3:
  `{enable_thinking: false}`), which otherwise consumes the token budget
  the plan has to fit behind.
- `output_location` — OPTIONAL S3 destination. When omitted, the envelope's
  item-level `output_location` is used. S3 outputs receive a unique
  run-and-round suffix. DB output locations are rejected: the server has no DB
  writer, so a DB destination cannot be produced.

A non-default `poll_interval` is REJECTED server-side (the server runs the
loop, so there is nothing to poll). Output-location writes are best-effort on
the server: a write failure is logged while the job can still be recorded as
completed, so `output_location` is not a durability guarantee.

A dynamic spec may declare a `library:` section: named, pre-configured op
templates the planner can emit by id. Each entry is a full op config (e.g. a
`DataRetrievalOp` with a real SQL template). The planner references one with
`{"ref": "<id>", "id": <unique>, "inputs": [...]}`; the template is
authoritative for its config, so the planner may only override `id` and
`inputs`. Ops without a `ref` are emitted fully inline. The planner sees the
library (id, op type, key config) in its system message each round.

The parent job's result carries the per-round plans. Missing, malformed,
unknown, or wrongly typed plan output fails the parent; it never becomes a
successful termination.

## Timeouts

The default HTTP timeout is **300 seconds**, set in `lumilake._base_client.DEFAULT_TIMEOUT`. Override it three ways:

1. **Client default** — pass `timeout=` to `LumilakeClient(...)` or `AsyncLumilakeClient(...)`. Applies to every call the client makes.
2. **Environment** — set `LUMILAKE_TIMEOUT=<seconds>` before constructing the client.
3. **Per call** — every resource method accepts `timeout=<seconds>` (or `request_timeout=` for poll-driven helpers like `wait` and `watch`). Long-running calls like `wait`, `watch`, `result`, and `artifact` are the usual reasons to bump it on a single request.

```python
client.jobs.result(job_id, timeout=900.0)
client.jobs.wait(job_id, timeout=1800.0, request_timeout=60.0)
client.jobs.artifact(job_id, path=..., output=..., timeout=600.0)
```

## Deploy Extra

Without the `deploy` extra, deploy methods except `init` raise `DeployError` with an install hint. Server API resources work without the deploy extra.

Deploy methods call `lumilake_deploy` directly. Async deploy methods dispatch the same Python calls through `asyncio.to_thread` so Docker and FlowMesh setup work does not block the event loop.
