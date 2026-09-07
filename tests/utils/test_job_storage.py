import threading
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from lumilake_server.utils.job_storage import InMemoryJobStorage


@dataclass
class _Record:
    job_id: str
    org_id: str
    user_id: str
    status: str
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    inputs: dict[str, object] = field(default_factory=dict)
    output_location: dict[str, object] = field(default_factory=dict)
    progress: dict[str, object] = field(default_factory=dict)
    result: dict[str, object] | None = None
    trace_ids: list[str] = field(default_factory=list)


def _record(job_id: str, status: str, submitted_at: str) -> _Record:
    return _Record(
        job_id=job_id,
        org_id="test-org",
        user_id="test-user",
        status=status,
        submitted_at=submitted_at,
        output_location={"graph": {"type": "s3", "prefix": "out.txt"}},
    )


def test_list_summaries_filters_by_status() -> None:
    storage = InMemoryJobStorage()
    storage.save(_record("job-1", "pending", "2026-02-20T00:00:00+00:00"))
    storage.save(_record("job-2", "completed", "2026-02-21T00:00:00+00:00"))
    storage.save(_record("job-3", "running", "2026-02-22T00:00:00+00:00"))

    items, total = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=None,
        statuses={"completed", "pending"},
        page=1,
        page_size=20,
    )

    assert total == 2
    assert [item["job_id"] for item in items] == ["job-2", "job-1"]


def test_list_summaries_paginates_descending_by_submission_time() -> None:
    storage = InMemoryJobStorage()
    storage.save(_record("job-1", "pending", "2026-02-20T00:00:00+00:00"))
    storage.save(_record("job-2", "pending", "2026-02-21T00:00:00+00:00"))
    storage.save(_record("job-3", "pending", "2026-02-22T00:00:00+00:00"))

    page_1, total = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=None,
        statuses=None,
        page=1,
        page_size=2,
    )
    page_2, _ = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=None,
        statuses=None,
        page=2,
        page_size=2,
    )

    assert total == 3
    assert [item["job_id"] for item in page_1] == ["job-3", "job-2"]
    assert [item["job_id"] for item in page_2] == ["job-1"]


def test_summary_validation_rejects_invalid_status() -> None:
    storage = InMemoryJobStorage()
    with pytest.raises(ValidationError):
        storage.save(_record("job-bad", "queued", "2026-02-20T00:00:00+00:00"))


def test_list_summaries_filters_by_job_ids_before_pagination() -> None:
    storage = InMemoryJobStorage()
    storage.save(_record("job-1", "pending", "2026-02-20T00:00:00+00:00"))
    storage.save(_record("job-2", "pending", "2026-02-21T00:00:00+00:00"))
    storage.save(_record("job-3", "pending", "2026-02-22T00:00:00+00:00"))

    page_1, total = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=frozenset({"job-1", "job-3"}),
        statuses=None,
        page=1,
        page_size=1,
    )
    page_2, _ = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=frozenset({"job-1", "job-3"}),
        statuses=None,
        page=2,
        page_size=1,
    )

    assert total == 2
    assert [item["job_id"] for item in page_1] == ["job-3"]
    assert [item["job_id"] for item in page_2] == ["job-1"]


def test_iter_summaries_filters_by_status() -> None:
    storage = InMemoryJobStorage()
    storage.save(_record("job-1", "pending", "2026-02-20T00:00:00+00:00"))
    storage.save(_record("job-2", "running", "2026-02-21T00:00:00+00:00"))
    storage.save(_record("job-3", "completed", "2026-02-22T00:00:00+00:00"))

    in_flight = {s.job_id for s in storage.iter_summaries({"pending", "running"})}
    assert in_flight == {"job-1", "job-2"}

    all_ids = {s.job_id for s in storage.iter_summaries()}
    assert all_ids == {"job-1", "job-2", "job-3"}


class _RaceDict(dict[str, threading.Lock]):
    """Lock map that holds both racers inside the fast-path lookup.

    ``_save_lock`` reads the map once before taking the guard and again inside
    it. Holding each thread at the barrier just after its *first* read forces
    both to see an empty map and then race the create-and-insert. Later reads
    must not block, or the thread holding the guard would wait forever on the
    thread blocked acquiring it.
    """

    def __init__(self, barrier: threading.Barrier) -> None:
        super().__init__()
        self._barrier = barrier
        self._arrived: set[int] = set()
        self._arrived_guard = threading.Lock()

    def get(  # type: ignore[override]
        self, key: str, default: "threading.Lock | None" = None
    ) -> "threading.Lock | None":
        ident = threading.get_ident()
        with self._arrived_guard:
            first_read = ident not in self._arrived
            self._arrived.add(ident)
        value = super().get(key, default)
        if first_read:
            # Block *after* reading, so neither thread's first read can observe
            # the other's insert.
            self._barrier.wait(timeout=5)
        return value


def test_save_lock_factory_returns_same_lock_under_race() -> None:
    """Two threads racing the first-use lock factory get the same lock."""
    storage = InMemoryJobStorage()
    job_id = "job-race"
    # NOTE: no pre-populating save for "job-race" — the first save must race
    # the lock factory.

    barrier = threading.Barrier(2)
    storage._save_locks = _RaceDict(barrier)

    locks: list[threading.Lock | None] = [None, None]
    errors: list[Exception | None] = [None, None]

    def _acquire(idx: int) -> None:
        try:
            locks[idx] = storage._save_lock(job_id)
        except Exception as exc:  # pragma: no cover
            errors[idx] = exc

    threads = [threading.Thread(target=_acquire, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    assert errors == [None, None]
    assert locks[0] is not None and locks[1] is not None
    assert locks[0] is locks[1]
