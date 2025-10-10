# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path

import imageio
import mediapy as media
import numpy as np
import torch
from mvbench import payload
from mvbench.data.base import CameraView
from mvbench.utils.camera_model import IdealPinholeCamera

from .base import BaseData, Calibration


class GeneratedData(BaseData):
    def __init__(self, traj_path: Path | None = None):
        self.traj_path = traj_path
        self.videos: dict[CameraView, np.ndarray] = {}
        if traj_path is not None:
            self.traj_pos = np.load(traj_path)
        else:
            self.traj_pos = None

    def __post_init__(self):
        assert len(self.videos) > 0, "No videos loaded"
        if self.traj_path is not None and self.traj_pos is not None:
            assert (
                self.traj_pos.shape[0] == self.num_frames()
            ), "Trajectory length mismatch"

    def get_image(self, view: CameraView, frame_idx: int) -> torch.Tensor:
        return torch.from_numpy(self.videos[view][frame_idx]) / 255.0

    def get_calibration(self) -> Calibration:
        cam_intrinsics, cam_extrinsics = payload.load_median_calib()

        # Filter to only include views we actually have
        filtered_intrinsics = {}
        filtered_extrinsics = {}

        for view in self.videos.keys():
            if view in cam_intrinsics:
                intr = cam_intrinsics[view]
                intr.rescale(self.videos[view].shape[2], self.videos[view].shape[1])
                filtered_intrinsics[view] = intr

            # Always add extrinsics for views we have (even if default)
            if view in cam_extrinsics:
                filtered_extrinsics[view] = cam_extrinsics[view]
            else:
                # Add identity transform if extrinsics not available
                filtered_extrinsics[view] = np.eye(4)

        rig_poses = np.eye(4)[None].repeat(self.num_frames(), axis=0)

        if self.traj_pos is not None:
            # We ignore extr provided by the calib since scale won't match anyway.
            rig_poses[:, :3, 3] = self.traj_pos

        return Calibration(
            intrinsics=filtered_intrinsics,
            extrinsics=filtered_extrinsics,
            rig_trajectory=rig_poses,
        )

    def num_frames(self) -> int:
        ex_key = list(self.videos.keys())[0]
        return self.videos[ex_key].shape[0]


class GeneratedStackedData(GeneratedData):
    def __init__(
        self,
        video_path: Path,
        layout: list[list[CameraView]],
        traj_path: Path | None = None,
    ):
        super().__init__(traj_path)

        self.video_path = video_path
        video = media.read_video(self.video_path)
        _, h, w, c = video.shape
        assert c == 3, "Only RGB videos are supported"

        num_rows = len(layout)
        num_cols = len(layout[0])
        vid_height = h // num_rows
        vid_width = w // num_cols

        for row_idx, row in enumerate(layout):
            assert len(row) == num_cols, "All rows must have the same length"
            for col_idx, view in enumerate(row):
                self.videos[view] = video[
                    :,
                    row_idx * vid_height : (row_idx + 1) * vid_height,
                    col_idx * vid_width : (col_idx + 1) * vid_width,
                    :,
                ]


class GeneratedSequentialData(GeneratedData):
    def __init__(
        self, video_path: Path, seq: list[CameraView], traj_path: Path | None = None
    ):
        super().__init__(traj_path)

        self.video_path = video_path
        video = media.read_video(self.video_path)
        n, h, w, c = video.shape
        assert c == 3, "Only RGB videos are supported"

        vid_length = n // len(seq)
        for idx, view in enumerate(seq):
            self.videos[view] = video[idx * vid_length : (idx + 1) * vid_length]


class GeneratedSeparateData(GeneratedData):
    def __init__(
        self, video_paths: dict[CameraView, Path], traj_path: Path | None = None
    ):
        super().__init__(traj_path)

        self.video_paths = video_paths
        for view, path in video_paths.items():
            self.videos[view] = media.read_video(path)


class RawNoCalibData(GeneratedData):
    def __init__(self, video_path: Path):
        super().__init__(None)

        self.video_path = video_path
        video = media.read_video(self.video_path)
        _, h, w, c = video.shape
        assert c == 3, "Only RGB videos are supported"

        self.videos[CameraView.RIG] = video

    def get_calibration(self) -> Calibration:
        return Calibration.null()


class RawNoCalibDataFromBytes(GeneratedData):
    """Modified from RawNoCalibData to read video from binary object instead of a filename."""

    def __init__(self, video_bytes: bytes):
        super().__init__(None)
        # Load video from binary and extract fps
        video = imageio.v3.imread(io.BytesIO(video_bytes), extension=".mp4")
        assert video.shape[-1] == 3, "Only RGB videos are supported"
        self.videos[CameraView.CAM0] = video

        # Extract fps from video metadata
        try:
            metadata = imageio.v3.immeta(io.BytesIO(video_bytes), extension=".mp4")
            self.fps = metadata.get("fps", 30)
        except Exception:  # noqa: BLE001
            # Fallback to default fps if extraction fails
            print("Could not extract fps from video, using default of 30")
            self.fps = 30

    def get_calibration(self) -> Calibration:
        return Calibration.null()


class RawDataFromBytesList(GeneratedData):
    """Data class for multiple videos from binary objects with dynamic camera calibration."""

    def __init__(
        self,
        video_bytes: dict[CameraView, bytes],
        camera_params: dict[CameraView, dict] = None,
    ):
        super().__init__(None)
        self.fps = 30  # Default fps
        self.camera_params = camera_params or {}

        # Load videos for each camera view
        for view, bytes_data in video_bytes.items():
            video = imageio.v3.imread(io.BytesIO(bytes_data), extension=".mp4")
            assert video.shape[-1] == 3, "Only RGB videos are supported"
            self.videos[view] = video

        # Extract fps from the first video
        first_bytes = next(iter(video_bytes.values()))
        try:
            metadata = imageio.v3.immeta(io.BytesIO(first_bytes), extension=".mp4")
            self.fps = metadata.get("fps", 30)
        except Exception:  # noqa: BLE001
            # Fallback to default fps if extraction fails
            print("Could not extract fps from video, using default of 30")
            self.fps = 30

    @staticmethod
    def _quaternion_to_rotation_matrix(
        qx: float, qy: float, qz: float, qw: float
    ) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix with automatic normalization.

        Args:
            qx, qy, qz, qw: Quaternion components (will be normalized internally)

        Returns:
            3x3 rotation matrix
        """
        # Normalize quaternion
        norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm > 0:
            qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        return np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ]
        )

    def get_frame_calibration(self, frame_idx: int) -> Calibration:
        """Return calibration for a specific frame with dynamic camera parameters.

        Args:
            frame_idx: Frame index for which to get calibration

        Returns:
            Calibration object with camera parameters for the specified frame
        """
        if not self.camera_params:
            return Calibration.null()

        # Create intrinsic camera objects for each view at this frame
        cam_intrinsics = {}
        cam_extrinsics = {}

        for view in [CameraView.CAM0, CameraView.CAM1]:
            if view in self.camera_params:
                camera_params = self.camera_params[view]
                intrinsics_list = camera_params["intrinsics"]
                extrinsics_list = camera_params["extrinsics"]

                if len(intrinsics_list) == 0 or frame_idx >= len(intrinsics_list):
                    continue

                # Use frame-specific parameters
                intrinsics = intrinsics_list[frame_idx]
                extrinsics = extrinsics_list[frame_idx]

                # Extract intrinsic parameters directly
                fx, fy, cx, cy = intrinsics

                # Get video dimensions for this view
                h, w, _ = self.videos[view][0].shape

                # Create IdealPinholeCamera object directly from parameters
                cam_intrinsics[view] = IdealPinholeCamera(
                    f_x=fx, f_y=fy, width=w, height=h
                )

                # Convert extrinsics (quaternion + translation) to world-to-camera matrix
                qx, qy, qz, qw, tx, ty, tz = extrinsics

                # Convert quaternion to rotation matrix (normalization handled internally)
                R = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)

                # Create 4x4 world-to-camera matrix
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = [tx, ty, tz]

                cam_extrinsics[view] = w2c

        # Create identity trajectory for the rig (no trajectory data available)
        rig_poses = np.eye(4)[None].repeat(self.num_frames(), axis=0)

        return Calibration(
            intrinsics=cam_intrinsics,
            extrinsics=cam_extrinsics,
            rig_trajectory=rig_poses,
        )

    def get_calibration(self) -> Calibration:
        """Return calibration using first frame parameters for backward compatibility."""
        return self.get_frame_calibration(0)
