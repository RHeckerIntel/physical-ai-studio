import asyncio
from multiprocessing.synchronize import Event as EventClass
from typing import Any

from control.environment_data_manifest import CameraManifestEntry, EnvironmentDataManifest, RobotManifestEntry
from robots.robot_client_factory import RobotClientFactory
from schemas.environment import EnvironmentWithRelations, TeleoperatorRobotWithRobot
from workers.camera_worker_registry import acquire_camera_worker, release_camera_worker
from workers.teleoperate_worker import TeleoperateWorker


class EnvironmentIntegration:
    """Responsible for setting up the workers for the robots and cameras in an environment."""

    manifest: EnvironmentDataManifest | None = None

    def __init__(
        self,
        environment: EnvironmentWithRelations,
        robot_client_factory: RobotClientFactory,
        mp_terminate_event: EventClass,
    ):
        self.robot = environment.robots[0]
        self.cameras = environment.cameras
        self.robot_client_factory = robot_client_factory
        self._mp_terminate_event = mp_terminate_event
        # Non-camera workers (e.g. TeleoperateWorker) owned exclusively by this instance.
        self._workers: list[Any] = []
        # Fingerprints of camera workers acquired from the shared registry.
        self._acquired_camera_fingerprints: list[str] = []

    async def setup_environment(self) -> None:
        try:
            follower = await self.robot_client_factory.build(self.robot.robot)
            features = follower.features()

            leader = None
            if (
                isinstance(self.robot.tele_operator, TeleoperatorRobotWithRobot)
                and self.robot.tele_operator.robot is not None
            ):
                leader = await self.robot_client_factory.build(self.robot.tele_operator.robot)

            teleoperate_worker = TeleoperateWorker(follower, leader, 100, self._mp_terminate_event)
            self._workers.append(teleoperate_worker)

            robot_entry = RobotManifestEntry(
                name=self.robot.robot.name,
                type=self.robot.robot.type,
                features=features,
                state=teleoperate_worker._output_state,
                actions=teleoperate_worker._output_actions,
                action_source=teleoperate_worker._action_source,
            )

            camera_entries = []
            for camera in self.cameras:
                worker = await acquire_camera_worker(camera, self._mp_terminate_event)
                self._acquired_camera_fingerprints.append(camera.fingerprint)
                camera_entries.append(
                    CameraManifestEntry(
                        id=str(camera.id),
                        name=camera.name,
                        width=worker._width,
                        height=worker._height,
                        frame_data=worker._frame_data,
                    )
                )

            for worker in self._workers:
                worker.start()

            for worker in self._workers:
                if hasattr(worker, "loaded_event"):
                    await asyncio.to_thread(worker.loaded_event.wait)

            self.manifest = EnvironmentDataManifest(robot=robot_entry, cameras=camera_entries)
        except Exception:
            for fingerprint in self._acquired_camera_fingerprints:
                release_camera_worker(fingerprint)
            self._acquired_camera_fingerprints.clear()
            for worker in self._workers:
                worker.stop()
            raise

    def teardown(self) -> None:
        for fingerprint in self._acquired_camera_fingerprints:
            release_camera_worker(fingerprint)
        self._acquired_camera_fingerprints.clear()
        for worker in self._workers:
            worker.stop()
