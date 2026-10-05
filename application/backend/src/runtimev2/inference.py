"""Load a model for a session, and check it fits.

A model sits above the environment for the same reason a dataset does: it is
chosen once and survives the devices being unloaded and loaded again. What it
has to agree with is the :class:`SessionShape` -- the cameras it names and the
joints its state vector was trained on.

The rate it is given is the rate actions are *emitted*, not the rate inference
runs. ``InferenceModel.select_action`` keeps the chunk it predicted and hands
back one action per call, so real inference happens every ``chunk_size`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from physicalai.inference.preprocessors import ToFloatTensorPreprocessor

from runtime.features import sanitize_camera_name
from runtimev2.control.model import check_camera_keys

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from physicalai.inference import InferenceModel

    from runtimev2.environment import SessionShape


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A policy ready to run."""

    model_id: UUID
    export_dir: Path
    chunk_size: int
    """Actions per inference. Inference runs every ``chunk_size`` emits.

    A property of the export, not a choice -- the rate actions are emitted at
    belongs to the control's config.
    """


def export_dir(models_dir: Path, model_id: UUID, backend: str) -> Path:
    """Return where a model's export for ``backend`` lives.

    The same layout the training and import paths write, which is the only
    definition of a model's location.
    """
    return models_dir / str(model_id) / "exports" / backend


def _ensure_image_conversion(model: InferenceModel) -> None:
    """Give a model that declares no preprocessing the conversion it needs.

    Cameras publish ``(H, W, 3)`` uint8; vision policies take channels-first
    float. Existing exports declare no preprocessors at all, so such a model
    fails on its first frame with a shape error from inside the network.

    Only when none was declared, so a manifest with its own pipeline stays
    authoritative.
    """
    if model.preprocessors:
        return
    logger.info("Export declares no preprocessing; supplying the frame-to-tensor conversion")
    model.preprocessors = [ToFloatTensorPreprocessor()]


def load_model(
    model_id: UUID,
    path: Path,
    shape: SessionShape,
    *,
    device: str,
) -> tuple[LoadedModel, InferenceModel]:
    """Load a policy and check it against this session's cameras.

    Blocking -- it reads an export and may initialise an accelerator -- so call
    it off the event loop.

    Returns:
        A ``(LoadedModel, InferenceModel)`` pair.

    Raises:
        FileNotFoundError: No export at ``path``.
        ModelCameraMismatchError: It names an image input no camera provides.
    """
    from physicalai.inference import InferenceModel

    if not path.exists():
        raise FileNotFoundError(str(path))
    model = InferenceModel(export_dir=path, policy_name=None, backend="auto", device=device)
    _ensure_image_conversion(model)
    check_camera_keys(model, [sanitize_camera_name(camera.name) for camera in shape.cameras])
    logger.info("Model {} loaded from {} with chunk size {}", model_id, path, model.chunk_size)
    return (
        LoadedModel(model_id=model_id, export_dir=path, chunk_size=int(model.chunk_size)),
        model,
    )
