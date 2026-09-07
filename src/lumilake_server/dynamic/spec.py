"""Pydantic models for a dynamic-workflow YAML spec and its loader.

A dynamic workflow is a YAML document describing a plaintext ``goal`` and the
driver settings. This module models that spec as pydantic so parsing and
validation are handled by pydantic, not hand-rolled.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)


class DriverSettings(BaseModel):
    """Driver settings for a dynamic workflow run."""

    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    max_tokens: StrictInt = Field(default=768, ge=1)
    temperature: StrictFloat = 0.4
    preview_width: StrictInt = Field(default=900, ge=1)
    poll_interval: StrictFloat = 2.0
    job_timeout: StrictFloat = Field(default=600.0, gt=0)
    max_rounds: StrictInt = Field(default=10, ge=1)
    max_nodes_per_round: StrictInt = Field(default=8, ge=1)
    threshold: StrictFloat | None = None
    chat_template_kwargs: dict[StrictStr, Any] | None = None
    output_location: dict[StrictStr, Any] | None = None


class DynamicSpec(BaseModel):
    """Top-level dynamic workflow spec."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["dynamic"] = "dynamic"
    name: StrictStr | None = None
    goal: StrictStr
    driver: DriverSettings
    library: dict[StrictStr, dict[StrictStr, Any]] | None = None


def load_spec(path: str | Path) -> DynamicSpec:
    """Load and validate a dynamic workflow YAML spec."""
    text = Path(path).read_text()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    try:
        return DynamicSpec.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid dynamic spec in {path}: {exc}") from exc
