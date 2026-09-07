"""Tests for the dynamic-workflow spec models."""

import pytest
from pydantic import ValidationError

from lumilake_server.dynamic.spec import DriverSettings, DynamicSpec, load_spec


def test_dynamic_spec_requires_goal_and_driver() -> None:
    spec = DynamicSpec(
        name="demo",
        goal="analyze market data",
        driver=DriverSettings(model="Qwen/Qwen3-8B"),
    )
    assert spec.goal == "analyze market data"
    assert spec.driver.model == "Qwen/Qwen3-8B"
    assert spec.driver.max_nodes_per_round == 8


def test_dynamic_spec_rejects_missing_goal() -> None:
    with pytest.raises(ValidationError):
        DynamicSpec.model_validate({"driver": {"model": "Qwen/Qwen3-8B"}})


def test_dynamic_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DynamicSpec.model_validate(
            {
                "goal": "g",
                "blocks": [],
                "driver": {"model": "Qwen/Qwen3-8B"},
            }
        )


def test_driver_settings_defaults() -> None:
    driver = DriverSettings(model="Qwen/Qwen3-8B")
    assert driver.max_tokens == 768
    assert driver.temperature == 0.4
    assert driver.preview_width == 900
    assert driver.max_rounds == 10
    assert driver.max_nodes_per_round == 8
    assert driver.threshold is None
    assert driver.output_location is None


def test_driver_settings_max_nodes_per_round() -> None:
    driver = DriverSettings(model="Qwen/Qwen3-8B", max_nodes_per_round=3)
    assert driver.max_nodes_per_round == 3


def test_driver_settings_rejects_zero_max_rounds() -> None:
    with pytest.raises(ValidationError):
        DriverSettings(model="Qwen/Qwen3-8B", max_rounds=0)


def test_driver_settings_rejects_zero_max_nodes_per_round() -> None:
    with pytest.raises(ValidationError):
        DriverSettings(model="Qwen/Qwen3-8B", max_nodes_per_round=0)


def test_dynamic_spec_accepts_library() -> None:
    spec = DynamicSpec(
        name="demo",
        goal="analyze market data",
        driver=DriverSettings(model="Qwen/Qwen3-8B"),
        library={
            "sector_market_cap": {
                "op": "DataRetrievalOp",
                "data_spec": {"type": "lumid", "mode": "sql"},
            }
        },
    )
    assert spec.library is not None
    assert spec.library["sector_market_cap"]["op"] == "DataRetrievalOp"


def test_dynamic_spec_library_defaults_none() -> None:
    spec = DynamicSpec(goal="g", driver=DriverSettings(model="Qwen/Qwen3-8B"))
    assert spec.library is None


def test_load_spec(tmp_path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text(
        "name: demo\n"
        "goal: analyze market data\n"
        "driver:\n"
        "  model: Qwen/Qwen3-8B\n"
        "  max_nodes_per_round: 4\n"
    )
    spec = load_spec(path)
    assert spec.name == "demo"
    assert spec.goal == "analyze market data"
    assert spec.driver.max_nodes_per_round == 4


def test_load_spec_invalid_yaml(tmp_path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text("not: [valid")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_spec(path)


def test_load_spec_missing_goal(tmp_path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text("driver:\n  model: Qwen/Qwen3-8B\n")
    with pytest.raises(ValueError, match="invalid dynamic spec"):
        load_spec(path)


def test_driver_settings_bound_policy_is_exact() -> None:
    """Pin the exact numeric lower-bound policy for DriverSettings, so a bound
    being relaxed (e.g. ge=1 -> ge=0) fails loudly."""
    from annotated_types import Ge, Gt

    expected: dict[str, tuple[str, int]] = {
        "max_tokens": ("ge", 1),
        "preview_width": ("ge", 1),
        "job_timeout": ("gt", 0),
        "max_rounds": ("ge", 1),
        "max_nodes_per_round": ("ge", 1),
    }

    actual: dict[str, tuple[str, object]] = {}
    for name, field in DriverSettings.model_fields.items():
        for meta in field.metadata:
            if isinstance(meta, Ge):
                actual[name] = ("ge", meta.ge)
            elif isinstance(meta, Gt):
                actual[name] = ("gt", meta.gt)

    assert (
        actual == expected
    ), f"DriverSettings bound policy differs; expected {expected}, got {actual}"

    for name, (kind, bound) in expected.items():
        violating = bound - 1 if kind == "ge" else bound
        with pytest.raises(ValidationError):
            DriverSettings.model_validate({"model": "Qwen/Qwen3-8B", name: violating})
