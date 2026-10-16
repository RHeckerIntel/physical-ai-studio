"""Runtime session fixtures, registered for every test by the root conftest."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from runtime_session.registry import RuntimeSessionRegistry
from runtime_session.session import RuntimeSession
from schemas.environment import EnvironmentWithRelations
from services.camera_claims import CameraClaimRegistry
from tests.workers.fake_robot import FakeRobot

FOLLOWER_ID = uuid4()
LEADER_ID = uuid4()
JOINTS = ["shoulder_pan", "gripper"]


def make_environment(*, leader: bool = True, cameras: int = 1) -> EnvironmentWithRelations:
    def robot(robot_id, name):
        return {
            "id": str(robot_id),
            "name": name,
            "type": "SO101_Follower",
            "payload": {"connection_string": "", "serial_number": name},
        }

    teleoperator = {"type": "robot", "robot_id": str(LEADER_ID), "robot": robot(LEADER_ID, "Leader")}
    return EnvironmentWithRelations.model_validate(
        {
            "id": str(uuid4()),
            "name": "Bench",
            "robots": [
                {"robot": robot(FOLLOWER_ID, "Follower"), "tele_operator": teleoperator if leader else {"type": "none"}}
            ],
            "cameras": [
                {
                    "id": str(uuid4()),
                    "driver": "usb_camera",
                    "name": f"Cam {index}",
                    "fingerprint": {"path": f"/dev/video{index}"},
                    "hardware_name": None,
                    "payload": {"width": 160, "height": 120, "fps": 30},
                }
                for index in range(cameras)
            ],
        }
    )


@pytest.fixture
def robot_factory(mock_robot_client_factory, tmp_path):
    """Builds a FakeRobot per robot row, each on its own pretend device."""

    async def build(robot, _port_finder):
        return FakeRobot(str(tmp_path / f"{robot.id}.jsonl"), JOINTS, [1.0, 2.0]), object()

    mock_robot_client_factory.build_robot_driver = AsyncMock(side_effect=build)
    return mock_robot_client_factory


@pytest.fixture
def fake_cameras():
    """SharedCameras that always have a fresh black frame."""

    def frame() -> MagicMock:
        latest = MagicMock()
        latest.data = np.zeros((120, 160, 3), dtype=np.uint8)
        latest.timestamp = time.monotonic()
        return latest

    with patch("workers.camera_worker.build_shared_camera") as build:
        build.return_value.read_latest.side_effect = frame
        yield build


@pytest.fixture
def registry() -> RuntimeSessionRegistry:
    return RuntimeSessionRegistry()


@pytest.fixture
def camera_claims() -> CameraClaimRegistry:
    return CameraClaimRegistry()


@pytest.fixture
def new_session(registry, camera_claims, robot_factory):
    sessions: list[RuntimeSession] = []

    def create() -> RuntimeSession:
        session = RuntimeSession(
            project_id=uuid4(),
            project_name="Alpha",
            registry=registry,
            camera_claims=camera_claims,
            robot_factory=robot_factory,
        )
        registry.add(session)
        sessions.append(session)
        return session

    return create
