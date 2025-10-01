"""
Minimal payload module for CSE/TSE evaluation.
Only includes median calibration loading functionality.
"""
from pathlib import Path

import numpy as np
from mvbench.data.base import CameraView
from mvbench.utils.camera_model import FThetaCamera


def load_median_calib() -> (
    tuple[dict[CameraView, FThetaCamera], dict[CameraView, np.ndarray]]
):
    """Load median camera calibration data for Alpamayo V2.2 dataset."""
    payload_data = np.load(
        Path(__file__).parent / "av22_median_calib.npz", allow_pickle=True
    )

    camera_intrinsics: dict[CameraView, FThetaCamera] = {}
    camera_extrinsics: dict[CameraView, np.ndarray] = {}

    for cv in CameraView:
        if (intr_key := f"intrinsics_{cv.value}") in payload_data:
            ftheta_params = payload_data[intr_key].item()
            camera_intrinsics[cv] = FThetaCamera(
                cx=ftheta_params["center"][0],
                cy=ftheta_params["center"][1],
                width=ftheta_params["width"],
                height=ftheta_params["height"],
                bw_poly=ftheta_params["bw_poly"],
            )

        if (extr_key := f"pose_{cv.value}") in payload_data:
            camera_extrinsics[cv] = payload_data[extr_key]

    return camera_intrinsics, camera_extrinsics
