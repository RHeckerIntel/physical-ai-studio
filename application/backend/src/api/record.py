import asyncio
import multiprocessing as mp
from queue import Empty
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket
from fastapi.responses import Response
from loguru import logger

from api.dependencies import (
    ModelRegistryDep,
    RecordingLockedCamerasDep,
    RobotCalibrationServiceDep,
    RobotConnectionManagerDep,
    get_scheduler_ws,
)
from core.scheduler import Scheduler
from robots.robot_client_factory import RobotClientFactory
from schemas import Dataset, Model
from schemas.environment import EnvironmentWithRelations
from workers.robot_control_worker import RobotControlWorker

router = APIRouter(prefix="/api/record")


@router.get("/robot_control/ws", tags=["WebSocket"], summary="Robot Control (WebSocket)", status_code=426)
async def robot_control_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


async def handle_incoming(
    websocket: WebSocket,
    process: RobotControlWorker,
    locked_camera_fingerprints: set[str],
) -> None:
    """Handle incoming messages for robot control."""
    try:
        while True:
            data = await websocket.receive_json("text")

            if data["event"] == "load_environment":
                payload = data.get("data", {})
                environment = EnvironmentWithRelations.model_validate(payload["environment"])
                locked_camera_fingerprints.clear()
                locked_camera_fingerprints.update(camera.fingerprint for camera in environment.cameras)

            process.input_queue.put(data)
            await asyncio.sleep(0.05)
    except Exception as e:
        logger.error(f"Incoming task stopped: {e}")
        logger.info("Except: disconnected!")


async def handle_outgoing(websocket: WebSocket, queue: mp.Queue) -> None:
    """Handle outgoing messages for robot control."""
    try:
        while True:
            try:
                loop = asyncio.get_running_loop()

                message = await loop.run_in_executor(None, queue.get)
                await websocket.send_json(message)
            except Empty:
                await asyncio.sleep(0.05)
    except Exception as e:
        logger.error(f"Outgoing task stopped: {e}")


@router.websocket("/robot_control/ws")
async def robot_control_websocket(
    websocket: WebSocket,
    robot_manager: RobotConnectionManagerDep,
    calibration_service: RobotCalibrationServiceDep,
    scheduler: Annotated[Scheduler, Depends(get_scheduler_ws)],
    model_registry: ModelRegistryDep,
    locked_camera_fingerprints: RecordingLockedCamerasDep,
) -> None:
    """Robot control websocket."""
    await websocket.accept()
    queue: mp.Queue = mp.Queue(maxsize=1)
    process = RobotControlWorker(
        stop_event=scheduler.mp_stop_event,
        robot_client_factory=RobotClientFactory(
            robot_manager=robot_manager,
            calibration_service=calibration_service,
        ),
        queue=queue,
        model_worker_registry=model_registry,
    )
    process.start()

    incoming_task = asyncio.create_task(handle_incoming(websocket, process, locked_camera_fingerprints))
    outgoing_task = asyncio.create_task(handle_outgoing(websocket, queue))

    try:
        _, pending = await asyncio.wait(
            {incoming_task, outgoing_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if process is not None:
            process.input_queue.put_nowait({"event": "disconnect"})
            process.join(10)

        queue.close()
    finally:
        locked_camera_fingerprints.clear()
    logger.info("websocket handling done...")
