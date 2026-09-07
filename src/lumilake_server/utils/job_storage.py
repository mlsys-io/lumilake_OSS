import datetime as dt
import json
import logging
import os
import sqlite3
import threading
import time
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any, Literal

from lumilake import envs
from pydantic import BaseModel, ConfigDict, ValidationError

from lumilake_server.utils import lumid_data_client


class ArchiveNotFound(Exception):
    """Raised by archive backends when a key is missing."""


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, dict):
        return {key: _normalize_payload(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_payload(val) for val in value]
    return value


class JobStorage:
    def __init__(self) -> None:
        self.logger = logging.getLogger("JobStorage")
        # One lock per job id, kept for the process lifetime. Evicting an entry
        # is unsafe without refcounting: a thread holding lock L could be
        # racing a thread that evicts L (unlocked) and a third that creates L2,
        # reintroducing the stale-write race. Bounded by the number of distinct
        # job ids seen, which is acceptable for the server's job volume.
        self._save_locks: dict[str, threading.Lock] = {}
        self._save_locks_guard = threading.Lock()

    def _save_lock(self, job_id: str) -> threading.Lock:
        """Return the per-job write lock, creating it on first use.

        ``save`` runs via ``asyncio.to_thread``, so this is a
        ``threading.Lock``. The guard lock serializes only the
        create-and-insert, so racing first saves for one job id share a lock.
        """
        lock = self._save_locks.get(job_id)
        if lock is None:
            with self._save_locks_guard:
                lock = self._save_locks.get(job_id)
                if lock is None:
                    lock = threading.Lock()
                    self._save_locks[job_id] = lock
        return lock

    def save(self, record: Any) -> None:
        """Persist ``record`` under its per-job write lock.

        Backends implement :meth:`_save_locked`; holding the lock here means
        every backend serializes writes for one job id, including any added
        later.
        """
        with self._save_lock(record.job_id):
            self._save_locked(record)

    @abstractmethod
    def _save_locked(self, record: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, job_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def reserve_output_location(self, location_key: str, job_id: str) -> None:
        raise NotImplementedError

    def release_output_location(self, location_key: str, job_id: str) -> None:
        raise NotImplementedError

    def save_artifact(
        self, job_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        raise NotImplementedError

    def get_artifact(self, job_id: str, filename: str) -> tuple[bytes, str]:
        raise NotImplementedError

    def list_summaries(
        self,
        *,
        org_id: str,
        user_id: str | None,
        job_ids: frozenset[str] | None,
        statuses: set[str] | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        raise NotImplementedError

    def iter_summaries(
        self, statuses: set[str] | None = None
    ) -> Iterable["JobSummary"]:
        raise NotImplementedError


class JobSummary(BaseModel):
    job_id: str
    org_id: str
    user_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    submitted_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    optimization_seconds: float | None = None
    selection_seconds: float | None = None
    clustering_seconds: float | None = None
    error: str | None = None
    parent_job_id: str | None = None

    model_config = ConfigDict(extra="forbid")


def _summary_from_payload(data: dict[str, Any]) -> JobSummary:
    payload = {
        "job_id": data["job_id"],
        "org_id": data["org_id"],
        "user_id": data["user_id"],
        "status": data["status"],
        "submitted_at": data["submitted_at"],
        "started_at": data["started_at"] if "started_at" in data else None,
        "finished_at": data["finished_at"] if "finished_at" in data else None,
        "optimization_seconds": data.get("optimization_seconds"),
        "selection_seconds": data.get("selection_seconds"),
        "clustering_seconds": data.get("clustering_seconds"),
        "error": data["error"] if "error" in data else None,
        "parent_job_id": data.get("parent_job_id"),
    }
    return JobSummary.model_validate(payload)


def _sort_summaries(summaries: list[JobSummary]) -> list[JobSummary]:
    return sorted(
        summaries,
        key=lambda item: (
            item.submitted_at,
            item.job_id,
        ),
        reverse=True,
    )


def _filter_summaries(
    summaries: list[JobSummary],
    *,
    org_id: str,
    user_id: str | None,
    job_ids: frozenset[str] | None,
    statuses: set[str] | None,
) -> list[JobSummary]:
    filtered: list[JobSummary] = []
    for summary in summaries:
        if summary.org_id != org_id:
            continue
        if user_id is not None and summary.user_id != user_id:
            continue
        if job_ids is not None and summary.job_id not in job_ids:
            continue
        if statuses and summary.status not in statuses:
            continue
        # Dynamic children are internal to their parent; hide them from the
        # user-facing listing.
        if summary.parent_job_id is not None:
            continue
        filtered.append(summary)
    return filtered


class InMemoryJobStorage(JobStorage):
    def __init__(self) -> None:
        super().__init__()
        self._storage: dict[str, dict[str, Any]] = {}
        self._inputs: dict[str, dict[str, Any]] = {}
        self._progress: dict[str, dict[str, Any]] = {}
        self._result: dict[str, dict[str, Any]] = {}
        self._output_index: dict[str, str] = {}
        self._artifacts: dict[str, dict[str, tuple[bytes, str]]] = {}
        self._summaries: dict[str, JobSummary] = {}

    def _save_locked(self, record: Any) -> None:
        data = _normalize_payload(asdict(record))
        record_data = dict(data)
        record_data.pop("inputs", None)
        record_data.pop("progress", None)
        record_data.pop("result", None)
        self._storage[record.job_id] = record_data
        self._inputs[record.job_id] = data.get("inputs", {})
        self._progress[record.job_id] = data.get("progress", {})
        if data.get("result") is not None:
            self._result[record.job_id] = data.get("result", {})
        self._summaries[record.job_id] = _summary_from_payload(data)
        self.logger.info("Saved job %s to in-memory storage", record.job_id)

    def load(self, job_id: str) -> dict[str, Any] | None:
        self.logger.info("Loading job %s from in-memory storage", job_id)
        record = self._storage.get(job_id)
        if record is None:
            return None
        data = dict(record)
        data["inputs"] = self._inputs.get(job_id, {})
        data["progress"] = self._progress.get(job_id, {})
        if job_id in self._result:
            data["result"] = self._result.get(job_id)
        return data

    def reserve_output_location(self, location_key: str, job_id: str) -> None:
        existing = self._output_index.get(location_key)
        if existing and existing != job_id:
            record = self._storage.get(existing)
            if record and record.get("status") == "completed":
                self._output_index[location_key] = job_id
                return
            raise ValueError(f"output location {location_key} already reserved")
        self._output_index[location_key] = job_id

    def release_output_location(self, location_key: str, job_id: str) -> None:
        existing = self._output_index.get(location_key)
        if existing == job_id:
            del self._output_index[location_key]

    def save_artifact(
        self, job_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        self._artifacts.setdefault(job_id, {})[filename] = (data, content_type)
        return f"memory://{job_id}/artifacts/{filename}"

    def get_artifact(self, job_id: str, filename: str) -> tuple[bytes, str]:
        job_artifacts = self._artifacts.get(job_id)
        if not job_artifacts or filename not in job_artifacts:
            raise KeyError(filename)
        return job_artifacts[filename]

    def list_summaries(
        self,
        *,
        org_id: str,
        user_id: str | None,
        job_ids: frozenset[str] | None,
        statuses: set[str] | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        items = list(self._summaries.values())
        filtered = _filter_summaries(
            items, org_id=org_id, user_id=user_id, job_ids=job_ids, statuses=statuses
        )
        sorted_items = _sort_summaries(filtered)
        total = len(sorted_items)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = [item.model_dump(mode="json") for item in sorted_items[start:end]]
        return page_items, total

    def iter_summaries(self, statuses: set[str] | None = None) -> Iterable[JobSummary]:
        for summary in self._summaries.values():
            if statuses is None or summary.status in statuses:
                yield summary


class PersistentJobStorage(JobStorage):
    """Archive backed by the lumid-data-app blob HTTP API."""

    def __init__(self) -> None:
        super().__init__()
        prefix = envs.S3_ARCHIVE_PREFIX
        assert prefix, "S3_ARCHIVE_PREFIX is not set"
        self.key_prefix = prefix.strip("/")
        # Serializes the jobs_index / output_index read-modify-write so the
        # storage layer is safe to call concurrently from multiple threads
        # without holding a higher-level lock through every HTTP round-trip.
        self._index_lock = threading.Lock()

    def _put_blob(self, key: str, body: bytes, content_type: str) -> None:
        lumid_data_client.put_blob(key, body, content_type)

    def _get_blob(self, key: str) -> tuple[bytes, str]:
        try:
            return lumid_data_client.get_blob(key)
        except lumid_data_client.BlobNotFound as exc:
            raise ArchiveNotFound(key) from exc

    def _object_name(self, job_id: str) -> str:
        if self.key_prefix:
            return f"{self.key_prefix}/{job_id}/record.json"
        return f"{job_id}/record.json"

    def _job_object_name(self, job_id: str, filename: str) -> str:
        if self.key_prefix:
            return f"{self.key_prefix}/{job_id}/{filename}"
        return f"{job_id}/{filename}"

    def _output_index_name(self) -> str:
        if self.key_prefix:
            return f"{self.key_prefix}/output_index.json"
        return "output_index.json"

    def _jobs_index_name(self) -> str:
        if self.key_prefix:
            return f"{self.key_prefix}/jobs_index.json"
        return "jobs_index.json"

    def _save_record_blobs(self, record: Any) -> tuple[JobSummary, str]:
        """Write the per-job blobs and return ``(summary, record_object_name)``.

        Split out of :meth:`save` so an alternative index backend can reuse the
        record/inputs/progress/result writes verbatim. These are per-job keys
        and are NOT what overflows the blob quota -- only the monolithic
        ``jobs_index.json`` is (see :class:`SqliteJobStorage`).
        """
        data = _normalize_payload(asdict(record))
        record_data = dict(data)
        record_data.pop("inputs", None)
        record_data.pop("progress", None)
        record_data.pop("result", None)
        record_body = json.dumps(record_data, ensure_ascii=False).encode("utf-8")
        obj_name = self._object_name(record.job_id)
        self._put_json(obj_name, record_body)
        inputs_body = json.dumps(data.get("inputs", {}), ensure_ascii=False).encode(
            "utf-8"
        )
        progress_body = json.dumps(data.get("progress", {}), ensure_ascii=False).encode(
            "utf-8"
        )
        self._put_json(self._job_object_name(record.job_id, "inputs.json"), inputs_body)
        self._put_json(
            self._job_object_name(record.job_id, "progress.json"), progress_body
        )
        if data.get("result") is not None:
            result_body = json.dumps(data.get("result", {}), ensure_ascii=False).encode(
                "utf-8"
            )
            self._put_json(
                self._job_object_name(record.job_id, "result.json"), result_body
            )
        return _summary_from_payload(data), obj_name

    def _save_locked(self, record: Any) -> None:
        summary, obj_name = self._save_record_blobs(record)
        index_name = self._jobs_index_name()
        with self._index_lock:
            index = self._get_json_optional(index_name) or {}
            index[record.job_id] = summary.model_dump(mode="json")
            self._put_json(
                index_name, json.dumps(index, ensure_ascii=False).encode("utf-8")
            )
        self.logger.info("Saved job %s to archive blob %s", record.job_id, obj_name)

    def load(self, job_id: str) -> dict[str, Any] | None:
        obj_name = self._object_name(job_id)
        try:
            data = self._get_json(obj_name)
        except ArchiveNotFound:
            return None
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to load job %s: %s", job_id, exc)
            return None
        data["inputs"] = (
            self._get_json_optional(self._job_object_name(job_id, "inputs.json")) or {}
        )
        data["progress"] = (
            self._get_json_optional(self._job_object_name(job_id, "progress.json"))
            or {}
        )
        result = self._get_json_optional(self._job_object_name(job_id, "result.json"))
        if result is not None:
            data["result"] = result
        return data

    def reserve_output_location(self, location_key: str, job_id: str) -> None:
        index_name = self._output_index_name()
        with self._index_lock:
            index = self._get_json_optional(index_name) or {}
            existing = index.get(location_key)
            if existing and existing != job_id:
                record = self.load(existing)
                if record and record.get("status") == "completed":
                    index[location_key] = job_id
                else:
                    raise ValueError(f"output location {location_key} already reserved")
            else:
                index[location_key] = job_id
            body = json.dumps(index, ensure_ascii=False).encode("utf-8")
            self._put_json(index_name, body)

    def release_output_location(self, location_key: str, job_id: str) -> None:
        index_name = self._output_index_name()
        with self._index_lock:
            index = self._get_json_optional(index_name)
            if index is None:
                return
            if index.get(location_key) != job_id:
                return
            del index[location_key]
            body = json.dumps(index, ensure_ascii=False).encode("utf-8")
            self._put_json(index_name, body)

    def save_artifact(
        self, job_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        object_name = self._job_object_name(job_id, f"artifacts/{filename}")
        self._put_blob(object_name, data, content_type)
        return object_name

    def get_artifact(self, job_id: str, filename: str) -> tuple[bytes, str]:
        object_name = self._job_object_name(job_id, f"artifacts/{filename}")
        try:
            return self._get_blob(object_name)
        except ArchiveNotFound as exc:
            raise KeyError(filename) from exc

    def _put_json(self, object_name: str, body: bytes) -> None:
        self._put_blob(object_name, body, "application/json")

    def _get_json(self, object_name: str) -> dict[str, Any]:
        body, _ = self._get_blob(object_name)
        return json.loads(body.decode("utf-8"))

    def _get_json_optional(self, object_name: str) -> dict[str, Any] | None:
        try:
            return self._get_json(object_name)
        except ArchiveNotFound:
            return None

    def list_summaries(
        self,
        *,
        org_id: str,
        user_id: str | None,
        job_ids: frozenset[str] | None,
        statuses: set[str] | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        index = self._get_json_optional(self._jobs_index_name()) or {}
        items: list[JobSummary] = []
        for value in index.values():
            if isinstance(value, dict):
                try:
                    items.append(JobSummary.model_validate(value))
                except ValidationError as exc:
                    self.logger.warning("Ignoring invalid job summary entry: %s", exc)
        filtered = _filter_summaries(
            items, org_id=org_id, user_id=user_id, job_ids=job_ids, statuses=statuses
        )
        sorted_items = _sort_summaries(filtered)
        total = len(sorted_items)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = [item.model_dump(mode="json") for item in sorted_items[start:end]]
        return page_items, total

    def iter_summaries(self, statuses: set[str] | None = None) -> Iterable[JobSummary]:
        index = self._get_json_optional(self._jobs_index_name()) or {}
        for value in index.values():
            if not isinstance(value, dict):
                continue
            try:
                summary = JobSummary.model_validate(value)
            except ValidationError as exc:
                self.logger.warning("Ignoring invalid job summary entry: %s", exc)
                continue
            if statuses is None or summary.status in statuses:
                yield summary


def _epoch(value: dt.datetime | None) -> float | None:
    """Stable epoch seconds for ordering.

    ``_sort_summaries`` sorts ``datetime`` objects directly, so the SQL ORDER BY
    has to reproduce that exactly. Naive values are read as UTC rather than local
    time so a container's TZ can never reorder history.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.timestamp()


class SqliteJobStorage(PersistentJobStorage):
    """Blob-backed job records with the two monolithic INDEXES in SQLite.

    # Why this exists

    ``PersistentJobStorage`` keeps every job summary in a single
    ``jobs_index.json`` blob and rewrites it in full on every ``save()`` --
    and ``save()`` runs on each status transition, not just on submit. So the
    write is O(n) in job count and the blob grows without bound until
    ``put_blob`` gets HTTP 413 from lumid-data, at which point EVERY submission
    fails. ``output_index.json`` has the identical shape and the identical
    fault; fixing only the jobs index would leave the second one armed.

    Both indexes become tables here: ``save()`` is a single-row UPSERT and
    ``list_summaries`` is an indexed query with real LIMIT/OFFSET, so neither
    write nor read scales with history size any more.

    # What deliberately does NOT move

    Job records, inputs/progress/result, and artifacts stay in blob storage.
    They are per-job keys, they never drove the 413, and moving them would mean
    migrating existing data for no benefit. Only the indexes move, so an
    existing archive keeps working unchanged.

    # Why SQLite and not Postgres

    Lumilake has no database dependency; ``sqlite3`` is stdlib, so this adds
    none. The deployment is already single-writer -- ``replicas: 1`` with
    ``Recreate`` -- and was ALREADY single-writer-only because ``_index_lock``
    is an in-process lock, so SQLite gives up nothing that exists today.

    If multi-replica Lumilake is ever wanted, this is the wrong backend and
    Postgres is right -- but that change is bigger than storage, because the
    in-process locking has to go too. Choosing SQLite here is deliberate, not
    a default.

    NOTE: the database file must live on a real block device. SQLite locking is
    unsafe on NFS, so moving the volume to RWX/NFS would silently break this.
    """

    _SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS job_summaries (
            job_id               TEXT PRIMARY KEY,
            org_id               TEXT NOT NULL,
            user_id              TEXT NOT NULL,
            status               TEXT NOT NULL,
            submitted_at         TEXT NOT NULL,
            submitted_at_ts      REAL NOT NULL,
            started_at           TEXT,
            finished_at          TEXT,
            optimization_seconds REAL,
            selection_seconds    REAL,
            clustering_seconds   REAL,
            error                TEXT,
            parent_job_id        TEXT
        )
        """,
        # Serves the list_summaries filter (org/user/status) and its sort in one
        # index; submitted_at_ts is the sort key, never the text column.
        """
        CREATE INDEX IF NOT EXISTS ix_job_summaries_query
            ON job_summaries (org_id, user_id, status, submitted_at_ts DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_job_summaries_sort
            ON job_summaries (submitted_at_ts DESC, job_id DESC)
        """,
        """
        CREATE TABLE IF NOT EXISTS output_locations (
            location_key TEXT PRIMARY KEY,
            job_id       TEXT NOT NULL
        )
        """,
    )

    def __init__(self, db_path: str | None = None) -> None:
        super().__init__()
        self._db_path = db_path or envs.LUMILAKE_JOB_INDEX_DB_PATH
        parent = os.path.dirname(self._db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # check_same_thread=False + the inherited _index_lock: the server calls
        # storage from several threads, and every write below is taken under
        # that lock, so the connection is never touched concurrently.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._index_lock:
            # WAL so a long list_summaries cannot block a save.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            for stmt in self._SCHEMA:
                self._conn.execute(stmt)
            # Existing databases predate the parent_job_id column; CREATE TABLE
            # IF NOT EXISTS will not add it, so migrate it in idempotently.
            columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(job_summaries)"
                ).fetchall()
            }
            if "parent_job_id" not in columns:
                self._conn.execute(
                    "ALTER TABLE job_summaries ADD COLUMN parent_job_id TEXT"
                )
            self._conn.commit()
        # None = never pruned. NOT 0.0: time.monotonic()'s origin is
        # arbitrary, so on a freshly-booted host `now - 0.0 < 3600` is true and
        # the first prune would be skipped for the first hour of uptime.
        self._last_prune: float | None = None
        self._backfill_if_empty()

    # ---- index writes -------------------------------------------------

    def _save_locked(self, record: Any) -> None:
        summary, obj_name = self._save_record_blobs(record)
        row = summary.model_dump(mode="json")
        with self._index_lock:
            self._conn.execute(
                """
                INSERT INTO job_summaries (
                    job_id, org_id, user_id, status, submitted_at,
                    submitted_at_ts, started_at, finished_at,
                    optimization_seconds, selection_seconds,
                    clustering_seconds, error, parent_job_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET
                    org_id=excluded.org_id,
                    user_id=excluded.user_id,
                    status=excluded.status,
                    submitted_at=excluded.submitted_at,
                    submitted_at_ts=excluded.submitted_at_ts,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    optimization_seconds=excluded.optimization_seconds,
                    selection_seconds=excluded.selection_seconds,
                    clustering_seconds=excluded.clustering_seconds,
                    error=excluded.error,
                    parent_job_id=excluded.parent_job_id
                """,
                (
                    summary.job_id,
                    summary.org_id,
                    summary.user_id,
                    summary.status,
                    row["submitted_at"],
                    _epoch(summary.submitted_at),
                    row.get("started_at"),
                    row.get("finished_at"),
                    summary.optimization_seconds,
                    summary.selection_seconds,
                    summary.clustering_seconds,
                    summary.error,
                    summary.parent_job_id,
                ),
            )
            self._conn.commit()
        self._maybe_prune()
        self.logger.info("Saved job %s to archive blob %s", record.job_id, obj_name)

    # ---- index reads --------------------------------------------------

    def _where(
        self,
        *,
        org_id: str,
        user_id: str | None,
        job_ids: frozenset[str] | None,
        statuses: set[str] | None,
    ) -> tuple[str, list[Any]]:
        clauses = ["org_id = ?"]
        params: list[Any] = [org_id]
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if job_ids is not None:
            if not job_ids:
                return "0", []  # empty set matches nothing, mirroring the blob path
            marks = ",".join("?" for _ in job_ids)
            clauses.append(f"job_id IN ({marks})")
            params.extend(sorted(job_ids))
        if statuses:
            marks = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({marks})")
            params.extend(sorted(statuses))
        # Dynamic children are internal to their parent; hide them from the
        # user-facing listing. This must live in _where so the COUNT(*) and the
        # page query agree on the same total.
        clauses.append("parent_job_id IS NULL")
        return " AND ".join(clauses), params

    def list_summaries(
        self,
        *,
        org_id: str,
        user_id: str | None,
        job_ids: frozenset[str] | None,
        statuses: set[str] | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        where, params = self._where(
            org_id=org_id, user_id=user_id, job_ids=job_ids, statuses=statuses
        )
        with self._index_lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM job_summaries WHERE {where}", params
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT * FROM job_summaries WHERE {where} "
                "ORDER BY submitted_at_ts DESC, job_id DESC LIMIT ? OFFSET ?",
                [*params, page_size, max(page - 1, 0) * page_size],
            ).fetchall()
        return [self._row_to_summary(r).model_dump(mode="json") for r in rows], total

    def iter_summaries(self, statuses: set[str] | None = None) -> Iterable[JobSummary]:
        sql = "SELECT * FROM job_summaries"
        params: list[Any] = []
        if statuses:
            marks = ",".join("?" for _ in statuses)
            sql += f" WHERE status IN ({marks})"
            params.extend(sorted(statuses))
        with self._index_lock:
            rows = self._conn.execute(sql, params).fetchall()
        for row in rows:
            try:
                yield self._row_to_summary(row)
            except ValidationError as exc:
                self.logger.warning("Ignoring invalid job summary row: %s", exc)

    def _row_to_summary(self, row: sqlite3.Row) -> JobSummary:
        return JobSummary.model_validate(
            {
                "job_id": row["job_id"],
                "org_id": row["org_id"],
                "user_id": row["user_id"],
                "status": row["status"],
                "submitted_at": row["submitted_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "optimization_seconds": row["optimization_seconds"],
                "selection_seconds": row["selection_seconds"],
                "clustering_seconds": row["clustering_seconds"],
                "error": row["error"],
                "parent_job_id": row["parent_job_id"],
            }
        )

    # ---- output locations ---------------------------------------------

    def reserve_output_location(self, location_key: str, job_id: str) -> None:
        with self._index_lock:
            row = self._conn.execute(
                "SELECT job_id FROM output_locations WHERE location_key = ?",
                (location_key,),
            ).fetchone()
            existing = row["job_id"] if row else None
        # load() re-enters blob storage, so it must not run under the lock.
        if existing and existing != job_id:
            record = self.load(existing)
            if not (record and record.get("status") == "completed"):
                raise ValueError(f"output location {location_key} already reserved")
        with self._index_lock:
            self._conn.execute(
                "INSERT INTO output_locations (location_key, job_id) VALUES (?,?) "
                "ON CONFLICT(location_key) DO UPDATE SET job_id=excluded.job_id",
                (location_key, job_id),
            )
            self._conn.commit()

    def release_output_location(self, location_key: str, job_id: str) -> None:
        with self._index_lock:
            self._conn.execute(
                "DELETE FROM output_locations WHERE location_key = ? AND job_id = ?",
                (location_key, job_id),
            )
            self._conn.commit()

    # ---- retention -----------------------------------------------------

    def _maybe_prune(self) -> None:
        """Prune terminal rows older than the retention window, at most hourly.

        Without this the table grows forever and we have only moved the ceiling
        rather than removed it. Only completed/failed/cancelled rows are ever
        deleted -- a pending or running job is kept regardless of age, so a
        stuck job can never be silently erased.
        """
        days = envs.LUMILAKE_JOB_INDEX_RETENTION_DAYS
        if days <= 0:
            return
        now = time.monotonic()
        if self._last_prune is not None and now - self._last_prune < 3600:
            return
        self._last_prune = now
        cutoff = (dt.datetime.now(dt.UTC) - dt.timedelta(days=days)).timestamp()
        with self._index_lock:
            cur = self._conn.execute(
                "DELETE FROM job_summaries WHERE submitted_at_ts < ? "
                "AND status IN ('completed','failed','cancelled')",
                (cutoff,),
            )
            self._conn.commit()
        if cur.rowcount:
            self.logger.info(
                "Pruned %d job summaries older than %d days", cur.rowcount, days
            )

    # ---- one-time backfill ---------------------------------------------

    def _backfill_if_empty(self) -> None:
        """Import the legacy blob indexes once, if this table is still empty.

        Best-effort: the blobs are left in place, so a rollback to the blob
        backend loses nothing. A missing or unreadable index is not fatal --
        a fresh deployment simply has no history to import.
        """
        with self._index_lock:
            already = self._conn.execute(
                "SELECT EXISTS(SELECT 1 FROM job_summaries)"
            ).fetchone()[0]
        if already:
            return
        try:
            index = self._get_json_optional(self._jobs_index_name()) or {}
        except Exception as exc:  # pragma: no cover - network/permission issues
            self.logger.warning("Job index backfill skipped: %s", exc)
            return
        imported = 0
        for value in index.values():
            if not isinstance(value, dict):
                continue
            try:
                summary = JobSummary.model_validate(value)
            except ValidationError as exc:
                self.logger.warning("Skipping invalid summary during backfill: %s", exc)
                continue
            row = summary.model_dump(mode="json")
            with self._index_lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO job_summaries (job_id, org_id, user_id,"
                    " status, submitted_at, submitted_at_ts, started_at, finished_at,"
                    " optimization_seconds, selection_seconds, clustering_seconds,"
                    " error, parent_job_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        summary.job_id,
                        summary.org_id,
                        summary.user_id,
                        summary.status,
                        row["submitted_at"],
                        _epoch(summary.submitted_at),
                        row.get("started_at"),
                        row.get("finished_at"),
                        summary.optimization_seconds,
                        summary.selection_seconds,
                        summary.clustering_seconds,
                        summary.error,
                        summary.parent_job_id,
                    ),
                )
            imported += 1
        try:
            outputs = self._get_json_optional(self._output_index_name()) or {}
        except Exception:  # pragma: no cover
            outputs = {}
        for location_key, owner in outputs.items():
            if isinstance(owner, str):
                with self._index_lock:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO output_locations"
                        " (location_key, job_id) VALUES (?,?)",
                        (location_key, owner),
                    )
        with self._index_lock:
            self._conn.commit()
        if imported:
            self.logger.info(
                "Backfilled %d job summaries and %d output locations into %s",
                imported,
                len(outputs),
                self._db_path,
            )


_job_storage: JobStorage | None = None


def get_job_storage() -> JobStorage:
    global _job_storage
    if _job_storage is None:
        # Default stays "blob" so no existing deployment changes behaviour on
        # upgrade; sqlite is opt-in via LUMILAKE_JOB_INDEX_BACKEND.
        if envs.LUMILAKE_JOB_INDEX_BACKEND == "sqlite":
            _job_storage = SqliteJobStorage()
        else:
            _job_storage = PersistentJobStorage()
    return _job_storage
