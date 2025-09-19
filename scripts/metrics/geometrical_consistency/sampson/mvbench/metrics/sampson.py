from enum import Enum
from logging import warning
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot
import numpy as np
import torch
from einops import rearrange
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd

from mvbench.data.base import BaseData, CameraView
from mvbench.utils.camera_model import FThetaCamera, IdealPinholeCamera
from mvbench.utils.geometry import compute_rectify_map, rectify_image, rectify_kp
from mvbench.utils.logging import pbar


class SampsonFundamentalMethod(Enum):
    CALIBRATED = "calibrated"
    KNOWN_INTRINSIC = "known_intrinsic"
    UNKNOWN_INTRINSIC = "unknown_intrinsic"


def homog(x: torch.Tensor):
    assert x.shape[1] == 2
    return torch.cat([x, torch.ones_like(x[:, 0:1])], dim=1)


def skew(x: torch.Tensor):
    a, b, c = x[0].item(), x[1].item(), x[2].item()
    return torch.tensor([[0, -c, b], [c, 0, -a], [-b, a, 0]]).to(x)


class BaseSampsonMetric:
    def __init__(self, target_intrinsic: IdealPinholeCamera | None) -> None:
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        self.matcher = LightGlue(features="superpoint").eval().cuda()

        self.target_intrinsic = target_intrinsic
        self.precomputed_rectify_maps: dict[CameraView, torch.Tensor] = {}

    @torch.no_grad()
    def match(
        self, image0: torch.Tensor, image1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image0 = rearrange(image0, "h w c -> c h w").cuda()
        image1 = rearrange(image1, "h w c -> c h w").cuda()

        feats0 = self.extractor.extract(image0)
        feats1 = self.extractor.extract(image1)

        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        match_inds = matches01["matches"]

        return (
            feats0["keypoints"][match_inds[:, 0]],
            feats1["keypoints"][match_inds[:, 1]],
        )

    def sampson_error(self, kp0: torch.Tensor, kp1: torch.Tensor, F: torch.Tensor):
        """
        First-order approximation of point to line distance in image-space.

        kp0: (N, 2)
        kp1: (N, 2)
        F: (3, 3)
        """
        kp0_h = homog(kp0)
        kp1_h = homog(kp1)
        a_error = torch.sum(kp0_h.T * (F @ kp1_h.T), dim=0)

        d_error0 = (F.T @ kp0_h.T) ** 2
        d_error1 = (F @ kp1_h.T) ** 2

        return a_error**2 / (d_error0[0] + d_error0[1] + d_error1[0] + d_error1[1])

    def fundamental_from_calibration(
        self,
        T01: torch.Tensor,
        k0: torch.Tensor,
        k1: torch.Tensor,
    ) -> torch.Tensor:
        """
        F: (3, 3) F = k0_inv^T E k1_inv, and E = [t]_x R which is T_0.inv * T_1
        """
        k0_inv = torch.inverse(k0)
        k1_inv = torch.inverse(k1)
        t, R = T01[:3, 3], T01[:3, :3]
        if torch.sum(torch.abs(t)) < 1e-6:
            warning("Translation is close to zero")
        E = skew(t) @ R
        F = k0_inv.T @ E @ k1_inv
        return F

    def fundamental_from_8_point(
        self, kp0: torch.Tensor, kp1: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply a 8-point method with opencv.
        """
        F, _ = cv2.findFundamentalMat(
            kp1.cpu().numpy(), kp0.cpu().numpy(), method=cv2.RANSAC
        )
        return torch.from_numpy(F).to(kp0.device).float()

    def fundamental_from_5_point(
        self,
        kp0: torch.Tensor,
        kp1: torch.Tensor,
        k0: torch.Tensor,
        k1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a 5-point method with opencv.
        """
        k0_inv = torch.inverse(k0)
        k1_inv = torch.inverse(k1)
        kp0_norm = torch.matmul(k0_inv, homog(kp0).T).T[:, :2]
        kp1_norm = torch.matmul(k1_inv, homog(kp1).T).T[:, :2]

        E, _ = cv2.findEssentialMat(
            kp1_norm.cpu().numpy(),
            kp0_norm.cpu().numpy(),
            np.eye(3),
            method=cv2.RANSAC,
        )
        E = torch.from_numpy(E).to(kp0.device).float()
        return k0_inv.T @ E @ k1_inv

    def save_visualization(
        self,
        path: Path,
        img0: torch.Tensor,
        img1: torch.Tensor,
        kp0_rec: torch.Tensor,
        kp1_rec: torch.Tensor,
        cv0: CameraView,
        cv1: CameraView,
        calib_intr: dict[CameraView, FThetaCamera],
        pixel_error: torch.Tensor,
        backend: str = "cv2",
    ):
        cmap = matplotlib.colormaps.get_cmap("viridis")

        if self.target_intrinsic is None:
            img0_rec = img0
            img1_rec = img1

        else:
            if cv0 not in self.precomputed_rectify_maps:
                self.precomputed_rectify_maps[cv0] = compute_rectify_map(
                    calib_intr[cv0], self.target_intrinsic
                )

            if cv1 not in self.precomputed_rectify_maps:
                self.precomputed_rectify_maps[cv1] = compute_rectify_map(
                    calib_intr[cv1], self.target_intrinsic
                )

            img0_rec = rectify_image(
                img0,
                calib_intr[cv0],
                self.target_intrinsic,
                precomputed_map=self.precomputed_rectify_maps[cv0],
            )
            img1_rec = rectify_image(
                img1,
                calib_intr[cv1],
                self.target_intrinsic,
                precomputed_map=self.precomputed_rectify_maps[cv1],
            )

        pixel_error_np = pixel_error.cpu().numpy()
        color = cmap(pixel_error_np / 20.0)[:, :3]

        if backend == "cv2":
            img_canvas = np.concatenate(
                [
                    (img0_rec.numpy() * 255).astype(np.uint8),
                    (img1_rec.numpy() * 255).astype(np.uint8),
                ],
                axis=1,
            ).copy()
            for i in range(len(kp0_rec)):
                cv2.circle(
                    img_canvas,
                    tuple(kp0_rec[i].int().cpu().numpy()),
                    4,
                    (color[i] * 255).astype(np.uint8).tolist(),
                    -1,
                    lineType=cv2.LINE_AA,
                )
                cv2.circle(
                    img_canvas,
                    tuple(kp1_rec[i].int().cpu().numpy() + [img0_rec.shape[1], 0]),
                    4,
                    (color[i] * 255).astype(np.uint8).tolist(),
                    -1,
                    lineType=cv2.LINE_AA,
                )
                cv2.line(
                    img_canvas,
                    tuple(kp0_rec[i].int().cpu().numpy()),
                    tuple(kp1_rec[i].int().cpu().numpy() + [img0_rec.shape[1], 0]),
                    (color[i] * 255).astype(np.uint8).tolist(),
                    1,
                    lineType=cv2.LINE_AA,
                )

            cv2.putText(
                img_canvas,
                f"Mean: {np.mean(pixel_error_np):.2f} Median: {np.median(pixel_error_np):.2f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(path), cv2.cvtColor(img_canvas, cv2.COLOR_RGB2BGR))

        else:

            viz2d.plot_images([img0_rec.permute(2, 0, 1), img1_rec.permute(2, 0, 1)])
            viz2d.plot_matches(kp0_rec, kp1_rec, color=color.tolist(), lw=0.5)
            viz2d.save_plot(path)

            matplotlib.pyplot.close("all")


class CrossViewSampsonMetric(BaseSampsonMetric):
    def __init__(
        self,
        target_intrinsic: IdealPinholeCamera,
        fundamental_method: SampsonFundamentalMethod,
        keep_ratio: float = 1.0,
        visualization_folder: Path | None = None,
    ) -> None:
        super().__init__(target_intrinsic)
        self.fundamental_method = fundamental_method
        self.keep_ratio = keep_ratio
        self.visualization_folder = visualization_folder

    def compute(self, data: BaseData, view_pair: tuple[CameraView, CameraView]):
        data_calib = data.get_calibration()

        cv0, cv1 = view_pair
        t01 = torch.from_numpy(
            np.linalg.inv(data_calib.extrinsics[cv0]) @ data_calib.extrinsics[cv1]
        ).float()
        k0_rec = k1_rec = torch.from_numpy(self.target_intrinsic.K)

        kp0_rec_all, kp1_rec_all = [], []
        for frame_idx in pbar(
            range(data.num_frames()), desc=f"Cross metric {cv0.value} - {cv1.value}"
        ):
            img0 = data.get_image(cv0, frame_idx)
            img1 = data.get_image(cv1, frame_idx)
            kp0_raw, kp1_raw = self.match(img0, img1)
            kp0_raw, kp1_raw = kp0_raw.cpu(), kp1_raw.cpu()

            # Perform rectification
            kp0_rec = rectify_kp(
                kp0_raw, data_calib.intrinsics[cv0], self.target_intrinsic
            )
            kp1_rec = rectify_kp(
                kp1_raw, data_calib.intrinsics[cv1], self.target_intrinsic
            )
            kp0_rec_all.append(kp0_rec)
            kp1_rec_all.append(kp1_rec)

        kp_count = torch.tensor([len(t) for t in kp0_rec_all])
        kp_end = torch.cumsum(kp_count, dim=0)
        kp_start = kp_end - kp_count
        kp0_rec = torch.cat(kp0_rec_all, dim=0)
        kp1_rec = torch.cat(kp1_rec_all, dim=0)

        if self.fundamental_method == SampsonFundamentalMethod.CALIBRATED:
            f_mat = self.fundamental_from_calibration(t01, k0_rec, k1_rec)

        elif self.fundamental_method == SampsonFundamentalMethod.KNOWN_INTRINSIC:
            f_mat = self.fundamental_from_5_point(kp0_rec, kp1_rec, k0_rec, k1_rec)

        elif self.fundamental_method == SampsonFundamentalMethod.UNKNOWN_INTRINSIC:
            f_mat = self.fundamental_from_8_point(kp0_rec, kp1_rec)

        else:
            raise ValueError("Unknown method")

        sampson_error = self.sampson_error(kp0_rec, kp1_rec, f_mat)
        pixel_error = torch.sqrt(sampson_error)

        # Apply keep_ratio filtering if needed
        if self.keep_ratio < 1.0:
            filtered_kp0_rec_all = []
            filtered_kp1_rec_all = []
            filtered_pixel_errors_all = []

            for i in range(len(kp_count)):
                s, e = kp_start[i], kp_end[i]
                frame_pixel_error = pixel_error[s:e]
                frame_kp0 = kp0_rec[s:e]
                frame_kp1 = kp1_rec[s:e]

                if len(frame_pixel_error) > 0:
                    # Filter to keep only top K% (lowest errors) for this frame
                    num_keep = max(1, int(len(frame_pixel_error) * self.keep_ratio))
                    _, top_indices = torch.topk(frame_pixel_error, num_keep, largest=False)
                    top_indices = torch.sort(top_indices)[0]  # Keep original order

                    filtered_kp0_rec_all.append(frame_kp0[top_indices])
                    filtered_kp1_rec_all.append(frame_kp1[top_indices])
                    filtered_pixel_errors_all.append(frame_pixel_error[top_indices])
                else:
                    filtered_kp0_rec_all.append(frame_kp0)
                    filtered_kp1_rec_all.append(frame_kp1)
                    filtered_pixel_errors_all.append(frame_pixel_error)

            # Update variables with filtered data
            kp_count = torch.tensor([len(t) for t in filtered_kp0_rec_all])
            kp_end = torch.cumsum(kp_count, dim=0)
            kp_start = kp_end - kp_count
            kp0_rec = torch.cat(filtered_kp0_rec_all, dim=0)
            kp1_rec = torch.cat(filtered_kp1_rec_all, dim=0)
            pixel_error = torch.cat(filtered_pixel_errors_all, dim=0)

        result_mean = torch.tensor(
            [
                torch.mean(pixel_error[kp_start[i] : kp_end[i]])
                for i in range(len(kp_count))
            ]
        )
        result_median = torch.tensor(
            [
                torch.median(pixel_error[kp_start[i] : kp_end[i]])
                for i in range(len(kp_count))
            ]
        )

        if self.visualization_folder is not None:
            viz_path = self.visualization_folder / f"cross_{cv0.value}_{cv1.value}"
            viz_path.mkdir(exist_ok=True, parents=True)

            for frame_idx in range(data.num_frames()):
                s, e = kp_start[frame_idx], kp_end[frame_idx]
                self.save_visualization(
                    viz_path / f"{frame_idx:04d}.png",
                    data.get_image(cv0, frame_idx),
                    data.get_image(cv1, frame_idx),
                    kp0_rec[s:e],
                    kp1_rec[s:e],
                    cv0,
                    cv1,
                    data_calib.intrinsics,
                    pixel_error[s:e],
                )

        return result_mean, result_median


class DynamicCrossViewSampsonMetric(BaseSampsonMetric):
    """Cross-view Sampson metric with dynamic per-frame calibration support.

    Assumes pinhole cameras - no rectification needed.
    """

    def __init__(
        self,
        target_intrinsic: IdealPinholeCamera,
        fundamental_method: SampsonFundamentalMethod,
        keep_ratio: float = 1.0,
        visualization_folder: Path | None = None,
    ) -> None:
        super().__init__(target_intrinsic)
        self.fundamental_method = fundamental_method
        self.keep_ratio = keep_ratio
        self.visualization_folder = visualization_folder

    def compute(self, data, view_pair: tuple[CameraView, CameraView]):
        """Compute cross-view Sampson metric with dynamic per-frame calibration.

        Args:
            data: Data object that supports get_frame_calibration(frame_idx)
            view_pair: Tuple of camera views to compare

        Returns:
            Tuple of (result_mean, result_median) tensors
        """
        cv0, cv1 = view_pair

        kp0_rec_all, kp1_rec_all = [], []
        for frame_idx in pbar(
            range(data.num_frames()), desc=f"Dynamic cross metric {cv0.value} - {cv1.value}"
        ):
            # Get frame-specific calibration
            assert hasattr(data, 'get_frame_calibration')
            frame_calib = data.get_frame_calibration(frame_idx)

            img0 = data.get_image(cv0, frame_idx)
            img1 = data.get_image(cv1, frame_idx)
            kp0_raw, kp1_raw = self.match(img0, img1)
            kp0_raw, kp1_raw = kp0_raw.cpu(), kp1_raw.cpu()

            # No rectification needed for pinhole cameras - use raw keypoints directly
            kp0_rec_all.append(kp0_raw)
            kp1_rec_all.append(kp1_raw)

        kp_count = torch.tensor([len(t) for t in kp0_rec_all])
        kp_end = torch.cumsum(kp_count, dim=0)
        kp_start = kp_end - kp_count
        kp0_rec = torch.cat(kp0_rec_all, dim=0)
        kp1_rec = torch.cat(kp1_rec_all, dim=0)

        # For dynamic calibration, compute fundamental matrix based on method
        if self.fundamental_method == SampsonFundamentalMethod.CALIBRATED:
            # Use first frame's calibration for fundamental matrix
            assert hasattr(data, 'get_frame_calibration')
            first_frame_calib = data.get_frame_calibration(0)

            t01 = torch.from_numpy(
                np.linalg.inv(first_frame_calib.extrinsics[cv0]) @ first_frame_calib.extrinsics[cv1]
            ).float()
            k0_rec = k1_rec = torch.from_numpy(self.target_intrinsic.K)
            f_mat = self.fundamental_from_calibration(t01, k0_rec, k1_rec)

        elif self.fundamental_method == SampsonFundamentalMethod.KNOWN_INTRINSIC:
            k0_rec = k1_rec = torch.from_numpy(self.target_intrinsic.K)
            f_mat = self.fundamental_from_5_point(kp0_rec, kp1_rec, k0_rec, k1_rec)

        elif self.fundamental_method == SampsonFundamentalMethod.UNKNOWN_INTRINSIC:
            f_mat = self.fundamental_from_8_point(kp0_rec, kp1_rec)

        else:
            raise ValueError("Unknown method")

        sampson_error = self.sampson_error(kp0_rec, kp1_rec, f_mat)
        pixel_error = torch.sqrt(sampson_error)

        # Apply keep_ratio filtering if needed
        if self.keep_ratio < 1.0:
            filtered_kp0_rec_all = []
            filtered_kp1_rec_all = []
            filtered_pixel_error_all = []

            for i in range(len(kp0_rec_all)):
                start_idx = kp_start[i]
                end_idx = kp_end[i]
                frame_pixel_error = pixel_error[start_idx:end_idx]
                frame_kp0_rec = kp0_rec_all[i]
                frame_kp1_rec = kp1_rec_all[i]

                # Keep top percentage of keypoints based on error
                num_keep = max(1, int(len(frame_pixel_error) * self.keep_ratio))
                _, indices = torch.topk(frame_pixel_error, num_keep, largest=False)

                filtered_kp0_rec_all.append(frame_kp0_rec[indices])
                filtered_kp1_rec_all.append(frame_kp1_rec[indices])
                filtered_pixel_error_all.append(frame_pixel_error[indices])

            kp0_rec_all = filtered_kp0_rec_all
            kp1_rec_all = filtered_kp1_rec_all
            pixel_error = torch.cat(filtered_pixel_error_all)

            # Recalculate indices after filtering
            kp_count = torch.tensor([len(t) for t in kp0_rec_all])
            kp_end = torch.cumsum(kp_count, dim=0)
            kp_start = kp_end - kp_count

        # Compute per-frame statistics
        result_mean = torch.tensor([
            torch.mean(pixel_error[kp_start[i]:kp_end[i]]) if kp_start[i] < kp_end[i] else torch.tensor(0.0)
            for i in range(len(kp_count))
        ])
        result_median = torch.tensor([
            torch.median(pixel_error[kp_start[i]:kp_end[i]]) if kp_start[i] < kp_end[i] else torch.tensor(0.0)
            for i in range(len(kp_count))
        ])

        # Visualization support (if needed)
        if self.visualization_folder is not None:
            viz_path = self.visualization_folder / f"dynamic_cross_{cv0.value}_{cv1.value}"
            viz_path.mkdir(exist_ok=True, parents=True)

            for frame_idx in range(data.num_frames()):
                if frame_idx < len(kp_start) and kp_start[frame_idx] < kp_end[frame_idx]:
                    s, e = kp_start[frame_idx], kp_end[frame_idx]
                    frame_calib = data.get_frame_calibration(frame_idx)
                    self.save_visualization(
                        viz_path / f"{frame_idx:04d}.png",
                        data.get_image(cv0, frame_idx),
                        data.get_image(cv1, frame_idx),
                        kp0_rec[s:e] if len(kp0_rec) > e else torch.empty(0, 2),
                        kp1_rec[s:e] if len(kp1_rec) > e else torch.empty(0, 2),
                        cv0,
                        cv1,
                        frame_calib.intrinsics,
                        pixel_error[s:e] if len(pixel_error) > e else torch.empty(0),
                    )

        return result_mean, result_median


class TemporalSampsonMetric(BaseSampsonMetric):
    def __init__(
        self,
        target_intrinsic: IdealPinholeCamera | None,
        fundamental_method: SampsonFundamentalMethod,
        keep_ratio: float = 1.0,
        visualization_folder: Path | None = None,
    ) -> None:
        super().__init__(target_intrinsic)
        self.fundamental_method = fundamental_method
        self.keep_ratio = keep_ratio
        self.visualization_folder = visualization_folder

    def compute(self, data: BaseData, cv: CameraView, frame_gap: int = 1):
        data_calib = data.get_calibration()
        mean_error, median_error = [], []

        cv_poses = data_calib.get_trajectory(cv)
        for frame_idx in pbar(
            range(data.num_frames() - frame_gap), desc=f"Temporal metric {cv.value}"
        ):
            img0 = data.get_image(cv, frame_idx)
            img1 = data.get_image(cv, frame_idx + frame_gap)
            kp0_raw, kp1_raw = self.match(img0, img1)
            kp0_raw, kp1_raw = kp0_raw.cpu(), kp1_raw.cpu()

            # Perform rectification
            if self.target_intrinsic is None or cv not in data_calib.intrinsics:
                kp0_rec = kp0_raw
                kp1_rec = kp1_raw

            else:
                kp0_rec = rectify_kp(
                    kp0_raw, data_calib.intrinsics[cv], self.target_intrinsic
                )
                kp1_rec = rectify_kp(
                    kp1_raw, data_calib.intrinsics[cv], self.target_intrinsic
                )

            if self.fundamental_method == SampsonFundamentalMethod.CALIBRATED:
                t_trans = torch.from_numpy(
                    np.linalg.inv(cv_poses[frame_idx]) @ cv_poses[frame_idx + frame_gap]
                ).float()
                k_rec = torch.from_numpy(self.target_intrinsic.K)
                f_mat = self.fundamental_from_calibration(t_trans, k_rec, k_rec)

            elif self.fundamental_method == SampsonFundamentalMethod.KNOWN_INTRINSIC:
                k_rec = torch.from_numpy(self.target_intrinsic.K)
                f_mat = self.fundamental_from_5_point(kp0_rec, kp1_rec, k_rec, k_rec)

            elif self.fundamental_method == SampsonFundamentalMethod.UNKNOWN_INTRINSIC:
                f_mat = self.fundamental_from_8_point(kp0_rec, kp1_rec)

            else:
                raise ValueError("Unknown method")

            sampson_error = self.sampson_error(kp0_rec, kp1_rec, f_mat)
            pixel_error = torch.sqrt(sampson_error)

            # Filter to keep only top K% (lowest errors)
            if self.keep_ratio < 1.0 and len(pixel_error) > 0:
                num_keep = max(1, int(len(pixel_error) * self.keep_ratio))
                _, top_indices = torch.topk(pixel_error, num_keep, largest=False)
                top_indices = torch.sort(top_indices)[0]  # Keep original order
                pixel_error = pixel_error[top_indices]
                kp0_rec = kp0_rec[top_indices]
                kp1_rec = kp1_rec[top_indices]

            mean_error.append(torch.mean(pixel_error))
            median_error.append(torch.median(pixel_error))

            if self.visualization_folder is not None:
                viz_path = self.visualization_folder / f"temporal_{cv.value}"
                viz_path.mkdir(exist_ok=True, parents=True)
                self.save_visualization(
                    viz_path / f"{frame_idx:04d}.png",
                    img0,
                    img1,
                    kp0_rec,
                    kp1_rec,
                    cv,
                    cv,
                    data_calib.intrinsics,
                    pixel_error,
                )

        return torch.tensor(mean_error), torch.tensor(median_error)
