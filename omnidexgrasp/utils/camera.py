"""📷 Camera utilities for intrinsic parameter computation."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class CameraIntrinsics:
    """📐 Camera intrinsic parameters."""

    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int

def dynamic_intrinsics(
    base: CameraIntrinsics, target_width: int, target_height: int
) -> CameraIntrinsics:
    """🔄 Scale intrinsics for different image size.

    Args:
        base: Original camera intrinsics.
        target_width: Target image width.
        target_height: Target image height.

    Returns:
        Scaled camera intrinsics for the new image size.
    """
    scale_x = target_width / base.width
    scale_y = target_height / base.height
    return CameraIntrinsics(
        fx=base.fx * scale_x,
        fy=base.fy * scale_y,
        ppx=base.ppx * scale_x,
        ppy=base.ppy * scale_y,
        width=target_width,
        height=target_height,
    )


def compute_focal(intrinsics: CameraIntrinsics) -> float:
    """📐 Compute average focal length.

    Args:
        intrinsics: Camera intrinsic parameters.

    Returns:
        Average of fx and fy as focal length.
    """
    return (intrinsics.fx + intrinsics.fy) / 2


def intrinsics_to_3x3(cam: CameraIntrinsics) -> list[list[float]]:
    """📐 Convert CameraIntrinsics to 3x3 matrix (nested list for JSON)."""
    return [
        [cam.fx, 0.0, cam.ppx],
        [0.0, cam.fy, cam.ppy],
        [0.0, 0.0, 1.0],
    ]


def pose_to_quat_json(pose_4x4: list[list[float]]) -> dict:
    """📐 Convert 4x4 pose to quat_xyzw + translation dict.

    Returns:
        {"quat_xyzw": [qx,qy,qz,qw], "translation": [tx,ty,tz]}
    """
    pose = np.array(pose_4x4)
    quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()
    translation = pose[:3, 3].tolist()
    return {"quat_xyzw": quat_xyzw, "translation": translation}
