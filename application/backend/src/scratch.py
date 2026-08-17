#!/usr/bin/env python3

from physicalai.config import export_config
from physicalai.runtime.events import TickEvent
from physicalai.runtime.events import LifecycleEvent
from robots.robot_client_factory import RobotClientFactory
from utils.serial_robot_tools import RobotConnectionManager
from physicalai.capture import UVCCamera
from physicalai.capture import RealSenseCamera
from db.engine import async_session
import asyncio
from uuid import UUID
from schemas.environment import EnvironmentWithRelations
from services.environment_service import EnvironmentService

from physicalai.runtime import RobotRuntime
from physicalai.robot import SO101
from physicalai.runtime.action_sources import ActionSource, TeleopSource

project_id = UUID("23eb0442-5f3f-4203-afb2-9d77f76cd85f")
environment_id = UUID("c67b0b16-01b2-4647-877c-c2cabbb83405")


session = async_session()

robot_client_factory = RobotClientFactory(RobotConnectionManager())

@export_config(class_path="physicalai.runtime.DatasetRecorder")
class DatasetRecorder:
    def __init__(self) -> None:
        pass

    def on_lifecycle(self, event: LifecycleEvent) -> None:
        if event.event == "start":
            print("Start Recording")
        elif event.event == "shutdown":
            print("Clear everything down on shutdown... WIP")

    def on_tick(self, event: TickEvent) -> None:
        print("on tick")
        print(f"action: {event.action_sent}")
        print(f"ss: {event.robot_state.state}")

    def flush(self) -> None:            # per run() == per episode
        print("flush...")

    def close(self) -> None:            # once, at disconnect()
        print("on disconnect")

async def main():
    environment = await EnvironmentService(session).get_environment_by_id(project_id, environment_id)
    print(environment)
    robot = await robot_client_factory.build(environment.robots[0].robot)
    leader = await robot_client_factory.build(environment.robots[0].tele_operator.robot)


    recorder = DatasetRecorder()
    runtime = RobotRuntime(
        fps=30,
        robot=robot._robot,
        action_source=TeleopSource(
            leader=leader._robot,
        ),
        callbacks=[
            recorder,
        ],
        cameras={
            "overview": RealSenseCamera(serial_number="323522062395"),
            "wrist": UVCCamera(device="/dev/video6", width=640, height=480),
        },
    )

    try:
        runtime.connect()

        print(runtime._bus)

        runtime.run(duration_s=10)

    finally:
        runtime.disconnect()





asyncio.run(main())
