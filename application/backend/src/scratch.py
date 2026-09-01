#!/usr/bin/env python3

from services import SystemService
from runtime.dataset_features import build_lerobot_dataset_features
from pathlib import Path

from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
def convert():
    from physicalai.policies import SmolVLA
    path = Path("/home/ronald/.local/share/physicalai/models/e56f9bdd-72dc-4aa3-a81a-eb0c0273695a")

    backends = ["torch", "openvino"]
    ckpt = SmolVLA.load_from_checkpoint(str(path / "model.ckpt"))
    for backend in backends:
        ckpt.export(path / "exports" / backend, backend="torch")


def features():
    folder = Path("/home/ronald/.local/share/physicalai/datasets/9b320666-9a42-49fc-b30b-1a08daf7abd0")


    dataset = InternalLeRobotDataset(folder)
    episode = dataset.find_episode(0)

    print(episode.source)
    #sources = dataset._dataset.hf_dataset["source"]
    #print(sources)

    #features = build_lerobot_dataset_features(
    #    joint_names=["gripper", "elbow"],
    #    camera_specs={
    #        "gripper": (480, 640, 3),
    #    },
    #)

    #print(features)






features()
