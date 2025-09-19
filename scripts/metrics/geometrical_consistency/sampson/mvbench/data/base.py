from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from mvbench.utils.camera_model import FThetaCamera


class CameraView(Enum):
    RIG = "rig"
    FRONT = "front"
    FRONT_TELE = "front_tele"  # Overlaps with FRONT.
    CROSS_LEFT = "cross_left"
    CROSS_RIGHT = "cross_right"
    REAR_LEFT = "rear_left"
    REAR_RIGHT = "rear_right"
    REAR_TELE = "rear_tele"
    CAM0 = "camera_0"
    CAM1 = "camera_1"


@dataclass
class Calibration:
    intrinsics: dict[CameraView, FThetaCamera]
    extrinsics: dict[CameraView, np.ndarray]
    rig_trajectory: np.ndarray

    def __post_init__(self):
        assert set(self.intrinsics.keys()) == set(self.extrinsics.keys())
        assert self.rig_trajectory.ndim == 3

    def get_trajectory(self, camera_view: CameraView) -> np.ndarray:
        if camera_view in (CameraView.RIG, CameraView.CAM0, CameraView.CAM1):
            return self.rig_trajectory
        return self.rig_trajectory @ self.extrinsics[camera_view]

    @classmethod
    def null(cls) -> "Calibration":
        return cls({}, {}, np.eye(4)[None])


class BaseData(ABC):
    @abstractmethod
    def num_frames(self) -> int: ...

    @abstractmethod
    def get_image(self, view: CameraView, frame_idx: int) -> torch.Tensor: ...

    @abstractmethod
    def get_calibration(self) -> Calibration: ...
