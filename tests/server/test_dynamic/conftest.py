import asyncio
import time
from typing import Any

import pytest


@pytest.fixture
def wait_for_inflight_child():
    """Wait until a child job is actually executing, and return its id.

    A child id appears in ``child_job_ids`` before its task is created, and a
    run may have already-finished children from earlier rounds, so neither "a
    child exists" nor ``child_job_ids[-1]`` identifies the one currently in
    flight.
    """

    async def _wait(job_routes: Any, job_id: str, timeout: float = 5.0) -> str:
        fake = job_routes._fake_runtime_server
        terminal = {"completed", "failed", "cancelled"}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            parent = job_routes.jobs[job_id]
            for child_id in parent.child_job_ids:
                child = job_routes.jobs.get(child_id)
                if (
                    child is not None
                    and child_id in fake.execute_calls
                    and child.status not in terminal
                ):
                    return child_id
            await asyncio.sleep(0.005)
        raise AssertionError(
            f"no in-flight child for {job_id} within {timeout}s; "
            f"children={job_routes.jobs[job_id].child_job_ids} "
            f"execute_calls={fake.execute_calls}"
        )

    return _wait
