"""A command carrying a request_id is answered by an ack bearing the same id.

Nothing else correlates a reply with the command that caused it: the event
stream is one-way, so a client watching ``state`` cannot tell its own save from
another's, nor a slow one from a failed one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import (
    get_camera_claim_registry,
    get_dataset_service,
    get_environment_service,
    get_project_service,
    get_robot_client_factory,
)
from main import app
from services.camera_claims import CameraClaimRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

PROJECT_ID = uuid4()
URL = f"/api/projects/{PROJECT_ID}/runtimev2/ws"


class _NoEnvironments:
    async def get_environment_by_id(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("no such environment")


class _StubProjectService:
    async def get_project_by_id(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(id=PROJECT_ID, name="Demo")


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_environment_service] = _NoEnvironments
    app.dependency_overrides[get_dataset_service] = lambda: object()
    app.dependency_overrides[get_robot_client_factory] = lambda: object()
    app.dependency_overrides[get_project_service] = _StubProjectService
    app.dependency_overrides[get_camera_claim_registry] = CameraClaimRegistry
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _drain_to(websocket: object, event: str, limit: int = 8) -> dict:
    """Return the next message of ``event``, skipping state and observations."""
    for _ in range(limit):
        message = websocket.receive_json()  # type: ignore[attr-defined]
        if message["event"] == event:
            return message
    raise AssertionError(f"no {event!r} within {limit} messages")


class TestAcks:
    def test_a_failed_command_acks_with_the_reason(self, client: TestClient) -> None:
        with client.websocket_connect(URL) as websocket:
            websocket.receive_json()  # the opening state
            websocket.send_json({"event": "load_environment", "environment_id": str(uuid4()), "request_id": "req-1"})

            ack = _drain_to(websocket, "ack")

        assert ack["data"]["request_id"] == "req-1"
        assert ack["data"]["ok"] is False
        assert "no such environment" in ack["data"]["error"]

    def test_a_failed_command_without_an_id_still_reports_an_error(self, client: TestClient) -> None:
        """Fire-and-forget keeps the behaviour it had."""
        with client.websocket_connect(URL) as websocket:
            websocket.receive_json()
            websocket.send_json({"event": "load_environment", "environment_id": str(uuid4())})

            message = _drain_to(websocket, "error")

        assert "no such environment" in message["message"]

    def test_an_unknown_command_is_refused_rather_than_left_hanging(self, client: TestClient) -> None:
        """A client awaiting a command this build lacks must not wait forever."""
        with client.websocket_connect(URL) as websocket:
            websocket.receive_json()
            websocket.send_json({"event": "teleport", "request_id": "req-2"})

            ack = _drain_to(websocket, "ack")

        assert ack["data"]["request_id"] == "req-2"
        assert ack["data"]["ok"] is False
        assert "teleport" in ack["data"]["error"]

    def test_an_unknown_command_with_no_id_is_ignored(self, client: TestClient) -> None:
        """This endpoint changes shape; a client ahead of it keeps its socket."""
        with client.websocket_connect(URL) as websocket:
            websocket.receive_json()
            websocket.send_json({"event": "teleport"})
            websocket.send_json({"event": "unload_environment", "request_id": "req-3"})

            ack = _drain_to(websocket, "ack")

        assert ack["data"]["request_id"] == "req-3"
        assert ack["data"]["ok"] is True

    def test_a_successful_command_acks_after_its_state(self, client: TestClient) -> None:
        """So a client holding the ack already has the resulting state."""
        with client.websocket_connect(URL) as websocket:
            websocket.receive_json()
            websocket.send_json({"event": "unload_environment", "request_id": "req-4"})

            events = []
            for _ in range(8):
                message = websocket.receive_json()
                events.append(message["event"])
                if message["event"] == "ack":
                    break

        assert "ack" in events
        assert "state" in events
        assert events.index("state") < events.index("ack")
