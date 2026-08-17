from physicalai.inference.manifest import OrderedTensorSpec
from physicalai.inference.manifest import CameraSpec
from pydantic import BaseModel, ConfigDict, Field


class DatasetSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    features: dict[str, OrderedTensorSpec] = Field(default_factory=dict)
    cameras: dict[str, CameraSpec] = Field(default_factory=dict)
