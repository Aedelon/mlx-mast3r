"""Pair viewer for stereo reconstruction.

Estimates camera poses from a stereo pair using PnP.

Adapted from dust3r/cloud_opt/pair_viewer.py for NumPy/OpenCV.

Copyright (c) 2025 Delanoe Pirard / Aedelon. Apache 2.0 License.
Original dust3r code: Copyright (C) 2024-present Naver Corporation. CC BY-NC-SA 4.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .focal import estimate_focal, make_intrinsics
from .geometry import geotrf, inv


@dataclass
class PairViewerResult:
    """Result of pair viewer reconstruction."""

    # Per-view data (2 views)
    pts3d: list[np.ndarray]  # 3D points for each view [(H,W,3), (H,W,3)]
    depths: list[np.ndarray]  # Depth maps [(H,W), (H,W)]
    confs: list[np.ndarray]  # Confidence maps [(H,W), (H,W)]
    imgs: list[np.ndarray]  # RGB images [(H,W,3), (H,W,3)]

    # Camera parameters
    focals: list[float]  # Focal lengths [f1, f2]
    pp: list[tuple[float, float]]  # Principal points [(cx1,cy1), (cx2,cy2)]
    im_poses: list[np.ndarray]  # Camera-to-world poses [4x4, 4x4]
    intrinsics: list[np.ndarray]  # Intrinsic matrices [3x3, 3x3]

    @property
    def n_views(self) -> int:
        return 2


class PairViewer:
    """Reconstruct a scene from a stereo pair.

    This is a simplified version that estimates poses directly using PnP,
    without iterative optimization.
    """

    def __init__(
        self,
        min_conf_thr: float = 1.5,
        focal_method: str = "weiszfeld",
        verbose: bool = False,
    ):
        """Initialize PairViewer.

        Args:
            min_conf_thr: Minimum confidence threshold.
            focal_method: Method for focal estimation ('median' or 'weiszfeld').
            verbose: Print debug info.
        """
        self.min_conf_thr = min_conf_thr
        self.focal_method = focal_method
        self.verbose = verbose

    def __call__(
        self,
        out1: dict,
        out2: dict,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> PairViewerResult:
        """Reconstruct scene from stereo pair outputs.

        Args:
            out1: Output from model.reconstruct() for view 1.
            out2: Output from model.reconstruct() for view 2.
            img1: RGB image 1 (H, W, 3).
            img2: RGB image 2 (H, W, 3).

        Returns:
            PairViewerResult with reconstructed scene.
        """
        # Extract predictions
        pts3d_1 = out1["pts3d"]  # 3D points in view1's frame
        pts3d_2 = out2["pts3d"]  # 3D points in view2's frame
        conf_1 = out1["conf"].squeeze()
        conf_2 = out2["conf"].squeeze()

        H, W = pts3d_1.shape[:2]
        pp = (W / 2, H / 2)

        # Estimate focals
        focal_1 = estimate_focal(pts3d_1, pp, method=self.focal_method)
        focal_2 = estimate_focal(pts3d_2, pp, method=self.focal_method)

        if self.verbose:
            print(f"Estimated focals: {focal_1:.1f}, {focal_2:.1f}")

        # Compute confidence scores for each view
        conf_score_1 = float(conf_1.mean() * conf_2.mean())
        conf_score_2 = float(conf_2.mean() * conf_1.mean())

        if self.verbose:
            print(f"Confidence scores: {conf_score_1:.3f}, {conf_score_2:.3f}")

        # Estimate relative pose using PnP
        # We try to find the pose of camera 2 in camera 1's frame
        pose_2in1 = self._estimate_pose_pnp(pts3d_1, pts3d_2, conf_1, conf_2, focal_1, focal_2, pp)

        # Choose the reference frame based on confidence
        if conf_score_1 >= conf_score_2:
            # Use camera 1 as reference (identity pose)
            im_poses = [np.eye(4, dtype=np.float32), pose_2in1]
            # pts3d already in camera 1's frame
            depth_1 = pts3d_1[:, :, 2]
            depth_2 = geotrf(inv(pose_2in1), pts3d_2)[:, :, 2]
        else:
            # Use camera 2 as reference
            pose_1in2 = inv(pose_2in1)
            im_poses = [pose_1in2, np.eye(4, dtype=np.float32)]
            # Transform pts3d to camera 2's frame
            depth_1 = geotrf(inv(pose_1in2), pts3d_1)[:, :, 2]
            depth_2 = pts3d_2[:, :, 2]

        # Create intrinsic matrices
        K1 = make_intrinsics(focal_1, pp, H, W)
        K2 = make_intrinsics(focal_2, pp, H, W)

        # Normalize images
        img1_uint8 = self._normalize_image(img1)
        img2_uint8 = self._normalize_image(img2)

        return PairViewerResult(
            pts3d=[pts3d_1, pts3d_2],
            depths=[depth_1, depth_2],
            confs=[conf_1, conf_2],
            imgs=[img1_uint8, img2_uint8],
            focals=[focal_1, focal_2],
            pp=[pp, pp],
            im_poses=im_poses,
            intrinsics=[K1, K2],
        )

    def _estimate_pose_pnp(
        self,
        pts3d_1: np.ndarray,
        pts3d_2: np.ndarray,
        conf_1: np.ndarray,
        conf_2: np.ndarray,
        focal_1: float,
        focal_2: float,
        pp: tuple[float, float],
    ) -> np.ndarray:
        """Estimate relative pose using PnP RANSAC.

        Args:
            pts3d_1: 3D points from view 1.
            pts3d_2: 3D points from view 2.
            conf_1: Confidence map for view 1.
            conf_2: Confidence map for view 2.
            focal_1: Focal length for view 1.
            focal_2: Focal length for view 2.
            pp: Principal point.

        Returns:
            4x4 pose matrix (camera 2 to world/camera 1).
        """
        H, W = pts3d_1.shape[:2]

        # Create pixel grid
        pixels = np.mgrid[:W, :H].T.astype(np.float32)  # (H, W, 2)

        # Create mask for valid points
        mask = (
            (conf_1 > self.min_conf_thr)
            & (conf_2 > self.min_conf_thr)
            & np.isfinite(pts3d_1).all(axis=-1)
            & np.isfinite(pts3d_2).all(axis=-1)
            & (pts3d_1[:, :, 2] > 0.1)
            & (pts3d_2[:, :, 2] > 0.1)
        )

        if mask.sum() < 10:
            if self.verbose:
                print("Not enough valid points for PnP, using identity pose")
            return np.eye(4, dtype=np.float32)

        # Use pts3d_1 as 3D points (world), pixels as 2D projections
        pts3d_world = pts3d_1[mask]
        pixels_2d = pixels[mask]

        # Intrinsic matrix for camera 2
        K = np.array([[focal_2, 0, pp[0]], [0, focal_2, pp[1]], [0, 0, 1]], dtype=np.float32)

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d_world,
                pixels_2d,
                K,
                None,
                iterationsCount=100,
                reprojectionError=5,
                flags=cv2.SOLVEPNP_SQPNP,
            )

            if not success:
                if self.verbose:
                    print("PnP failed, using identity pose")
                return np.eye(4, dtype=np.float32)

            if self.verbose:
                print(f"PnP inliers: {len(inliers)}/{len(pts3d_world)}")

            # Convert to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Build world-to-cam pose
            pose_w2c = np.eye(4, dtype=np.float32)
            pose_w2c[:3, :3] = R
            pose_w2c[:3, 3] = tvec.flatten()

            # Invert to get cam-to-world
            pose_c2w = inv(pose_w2c)

            return pose_c2w

        except Exception as e:
            if self.verbose:
                print(f"PnP error: {e}")
            return np.eye(4, dtype=np.float32)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to uint8."""
        if img.dtype == np.uint8:
            return img
        if img.max() <= 1.0:
            return (img * 255).astype(np.uint8)
        return img.astype(np.uint8)
