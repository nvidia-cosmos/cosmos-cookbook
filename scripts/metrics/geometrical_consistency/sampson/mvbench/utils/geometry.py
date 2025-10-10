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

from pathlib import Path

import numpy as np
import torch

from .camera_model import FThetaCamera, IdealPinholeCamera


def compute_rectify_map(ftheta_cam: FThetaCamera, target_cam: IdealPinholeCamera):
    ys, xs = np.mgrid[0 : target_cam.height, 0 : target_cam.width]
    pixels = np.stack((xs, ys), axis=2)
    pinhole_cam = target_cam
    pinhole_rays = pinhole_cam.pixel2ray(pixels.reshape(-1, 2))  # hw x 3

    pos = ftheta_cam.ray2pixel(pinhole_rays)
    pos_norm = (
        2.0
        * pos
        / np.array(
            [ftheta_cam.width - 1.0, ftheta_cam.height - 1.0],
            dtype=np.float32,
        )
        - 1.0
    )
    pos_norm = torch.from_numpy(pos_norm).reshape(
        1, target_cam.height, target_cam.width, 2
    )
    return pos_norm


def rectify_image(
    image: torch.Tensor,
    ftheta_cam: FThetaCamera,
    target_cam: IdealPinholeCamera,
    precomputed_map: torch.Tensor | None = None,
) -> torch.Tensor:
    if precomputed_map is None:
        pos_norm = compute_rectify_map(ftheta_cam, target_cam)
    else:
        pos_norm = precomputed_map

    img = image.permute(2, 0, 1).float().unsqueeze(0)
    img = torch.nn.functional.grid_sample(
        img,
        pos_norm,
        mode="bilinear",
        align_corners=False,
    )[0].float()
    img = img.permute(1, 2, 0)
    return img


def rectify_kp(
    kp: torch.Tensor, ftheta_cam: FThetaCamera, target_cam: IdealPinholeCamera
) -> torch.Tensor:
    r, _ = ftheta_cam.pixel2ray(kp.cpu().numpy())
    kp_rec, _ = target_cam.ray2pixel(r)
    return torch.from_numpy(kp_rec).to(kp.device)
