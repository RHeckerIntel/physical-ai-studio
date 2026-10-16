"""Runtime session websocket messages and the session listing.

Client to server: one JSON command per message, each with a ``request_id``
that its ``ack`` echoes. Server to client: ``ack``, ``state`` on change and
``error`` as JSON text, and ``observation`` at the sampling rate as one binary
message per tick: ``[u32 big-endian header length][ObservationMessage JSON][jpeg bytes]``,
the JPEGs concatenated in the order of ``frames``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID  # noqa: TC003 - pydantic resolves annotations at runtime

from pydantic import BaseModel, Field

from schemas.base import UTCDatetime
from schemas.environment import CameraEnvironmentConfiguration, RobotEnvironmentConfiguration


class EnvironmentDraft(BaseModel):
    """An environment to load, saved or still being edited in the form."""

    name: str = "Unsaved environment"
    robots: list[RobotEnvironmentConfiguration]
    cameras: list[CameraEnvironmentConfiguration] = Field(default_factory=list)


class LoadEnvironmentCommand(BaseModel):
    command: Literal["load_environment"] = "load_environment"
    request_id: str
    environment: EnvironmentDraft


class UnloadEnvironmentCommand(BaseModel):
    command: Literal["unload_environment"] = "unload_environment"
    request_id: str


RuntimeSessionCommand = Annotated[
    LoadEnvironmentCommand | UnloadEnvironmentCommand,
    Field(discriminator="command"),
]


class SessionError(BaseModel):
    message: str
    error_code: str


class AckMessage(BaseModel):
    type: Literal["ack"] = "ack"
    request_id: str
    ok: bool
    error: SessionError | None = None
    result: dict[str, Any] | None = None


class SessionCamera(BaseModel):
    camera_id: UUID
    key: str
    """The store and frame key; also the key a dataset records this camera under."""
    height: int
    width: int


class SessionEnvironment(BaseModel):
    name: str
    robot_id: UUID
    leader_id: UUID | None
    joint_names: list[str]
    cameras: list[SessionCamera]


class SessionState(BaseModel):
    session_id: str
    environment: SessionEnvironment | None = None


class StateMessage(BaseModel):
    type: Literal["state"] = "state"
    data: SessionState


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    component: str
    message: str
    error_code: str
    session_alive: bool
    """False when the session has ended; the socket closes after this message."""


class ObservationFrame(BaseModel):
    key: str
    timestamp: float
    size: int
    """Bytes of this JPEG in the message body."""


class ObservationMessage(BaseModel):
    """The header of a binary observation message; ``frames`` lists only frames new since the last one."""

    type: Literal["observation"] = "observation"
    timestamp: float
    state: dict[str, float] | None
    action: dict[str, float] | None
    frames: list[ObservationFrame]


RuntimeSessionServerMessage = Annotated[
    AckMessage | StateMessage | ErrorMessage | ObservationMessage,
    Field(discriminator="type"),
]


class RuntimeSessionInfo(BaseModel):
    """One live runtime session, for the footer and the robot list."""

    session_id: str
    project_id: UUID
    project_name: str
    started_at: UTCDatetime
    environment_name: str | None
    robot_ids: list[UUID]
    camera_ids: list[UUID]


class RuntimeSessionsMessage(BaseModel):
    """Sent by ``/api/runtime/sessions/ws`` on connect and whenever the list changes."""

    sessions: list[RuntimeSessionInfo]
