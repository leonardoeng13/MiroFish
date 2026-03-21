"""
Pydantic-based request validators for the MiroFish API.
========================================================

Each model validates the JSON body of a specific endpoint so that
malformed inputs are rejected early with a clear 400 error before
reaching business logic.

Adding a new validator
-----------------------
1. Subclass :class:`pydantic.BaseModel`.
2. Add :class:`pydantic.Field` descriptors with ``min_length`` / ``ge`` / ``le``
   constraints as appropriate.
3. Optionally add a ``@field_validator`` for cross-field logic.
4. Call :func:`parse_request` in the route handler and return 400 on error.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator


# ---------------------------------------------------------------------------
# Graph API
# ---------------------------------------------------------------------------

class BuildGraphRequest(BaseModel):
    """Request body for POST /api/graph/build"""

    project_id: str = Field(..., min_length=1, description="Project ID")
    chunk_size: Optional[int] = Field(
        default=None, ge=100, le=10000, description="Text chunk size"
    )
    chunk_overlap: Optional[int] = Field(
        default=None, ge=0, le=500, description="Chunk overlap"
    )

    @field_validator("project_id")
    @classmethod
    def project_id_must_be_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("project_id must not be blank")
        return v


# ---------------------------------------------------------------------------
# Simulation API
# ---------------------------------------------------------------------------

class CreateSimulationRequest(BaseModel):
    """Request body for POST /api/simulation/create"""

    project_id: str = Field(..., min_length=1, description="Project ID")
    graph_id: Optional[str] = Field(default=None, description="Graph ID (optional)")
    enable_twitter: bool = Field(default=True)
    enable_reddit: bool = Field(default=True)

    @field_validator("project_id", "graph_id", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        if isinstance(v, str):
            return v.strip() or None
        return v


class PrepareSimulationRequest(BaseModel):
    """Request body for POST /api/simulation/prepare"""

    simulation_id: str = Field(..., min_length=1, description="Simulation ID")
    force_regenerate: bool = Field(default=False)


class RunSimulationRequest(BaseModel):
    """Request body for POST /api/simulation/run"""

    simulation_id: str = Field(..., min_length=1, description="Simulation ID")


class InterviewAgentRequest(BaseModel):
    """Request body for POST /api/simulation/interview"""

    simulation_id: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1, description="Agent name to interview")
    prompt: str = Field(
        ..., min_length=1, max_length=2000, description="Interview question"
    )


# ---------------------------------------------------------------------------
# Report API
# ---------------------------------------------------------------------------

class GenerateReportRequest(BaseModel):
    """Request body for POST /api/report/generate"""

    simulation_id: str = Field(..., min_length=1, description="Simulation ID")
    force_regenerate: bool = Field(default=False)


class ReportInteractRequest(BaseModel):
    """Request body for POST /api/report/interact"""

    simulation_id: str = Field(..., min_length=1)
    message: str = Field(
        ..., min_length=1, max_length=4000, description="Message for the report agent"
    )
    conversation_id: Optional[str] = Field(
        default=None, description="Existing conversation ID (optional)"
    )


def parse_request(model_class, data: dict):
    """
    Validate *data* against *model_class*.

    Returns ``(instance, None)`` on success or ``(None, error_message)`` on
    validation failure so callers can do a simple two-value unpack.

    Only :class:`pydantic.ValidationError` is caught; other exceptions
    (programming errors, unexpected types, etc.) are re-raised so they
    fail loudly rather than being silently converted to a 400 response.
    """
    try:
        return model_class(**data), None
    except ValidationError as exc:
        return None, str(exc)
