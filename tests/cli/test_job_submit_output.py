"""CLI coverage for lumilake job submit output-location handling.

These tests assert that the S3 and DB paths build the correct
output_location payloads, that each location type validates its required
fields locally, and that unsupported output types fail locally.
"""

from pathlib import Path
from typing import Any

import pytest
from lumilake_cli.commands import job as job_cmd
from lumilake_cli.commands.job import app
from typer.testing import CliRunner


class _StubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _CapturingClient:
    """Records the submitted payload and returns a canned success response."""

    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.base_url = "http://test"
        self.api_key = None
        self.timeout = 10.0

    def post(self, path: str, **kwargs: Any) -> _StubResponse:
        self.payload = kwargs.get("json")
        return _StubResponse(
            {"ok": True, "data": {"job_id": "j-1", "status": "completed"}}
        )


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_workflow(tmp_path: Path) -> Path:
    wf = tmp_path / "wf.yaml"
    wf.write_text("nodes: []")
    return wf


def test_submit_s3_builds_output_location(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _CapturingClient()
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)
    wf = _make_workflow(tmp_path)

    result = runner.invoke(
        app,
        [
            "submit",
            str(wf),
            "--format",
            "yaml",
            "--input",
            "sym=NVDA",
            "--output-prefix",
            "out/prefix/",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert client.payload is not None
    item = client.payload["data"][0]
    assert item["output_location"] == {"type": "s3", "prefix": "out/prefix/"}


def test_submit_missing_output_prefix_is_local_error(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _CapturingClient()
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)
    wf = _make_workflow(tmp_path)

    result = runner.invoke(
        app,
        [
            "submit",
            str(wf),
            "--format",
            "yaml",
            "--input",
            "sym=NVDA",
        ],
    )
    assert result.exit_code == 1
    assert "--output-prefix is required" in result.stderr
    # No HTTP request was attempted.
    assert client.payload is None


def test_submit_explicit_output_type_s3_builds_output_location(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Backward compat: the shipped ``--output-type s3`` must still work."""
    client = _CapturingClient()
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)
    wf = _make_workflow(tmp_path)

    result = runner.invoke(
        app,
        [
            "submit",
            str(wf),
            "--format",
            "yaml",
            "--input",
            "sym=NVDA",
            "--output-type",
            "s3",
            "--output-prefix",
            "out/prefix/",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert client.payload is not None
    item = client.payload["data"][0]
    assert item["output_location"] == {"type": "s3", "prefix": "out/prefix/"}


def test_submit_output_type_db_builds_output_location(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _CapturingClient()
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)
    wf = _make_workflow(tmp_path)

    result = runner.invoke(
        app,
        [
            "submit",
            str(wf),
            "--format",
            "yaml",
            "--input",
            "sym=NVDA",
            "--output-type",
            "db",
            "--output-table",
            "metrics",
            "--output-column",
            "score",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert client.payload is not None
    item = client.payload["data"][0]
    assert item["output_location"] == {
        "type": "db",
        "table": "metrics",
        "column": "score",
    }


def test_submit_db_missing_table_or_column_is_local_error(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _CapturingClient()
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)
    wf = _make_workflow(tmp_path)

    result = runner.invoke(
        app,
        [
            "submit",
            str(wf),
            "--format",
            "yaml",
            "--input",
            "sym=NVDA",
            "--output-type",
            "db",
            "--output-table",
            "metrics",
        ],
    )
    assert result.exit_code == 1
    assert "--output-table and --output-column required" in result.stderr
    assert client.payload is None


def test_submit_output_type_unknown_fails_locally(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _CapturingClient()
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)
    wf = _make_workflow(tmp_path)

    result = runner.invoke(
        app,
        [
            "submit",
            str(wf),
            "--format",
            "yaml",
            "--input",
            "sym=NVDA",
            "--output-type",
            "gcs",
            "--output-prefix",
            "ignored/",
        ],
    )
    assert result.exit_code == 1
    assert "Unsupported output type: gcs" in result.stderr
    # No HTTP request was attempted.
    assert client.payload is None
