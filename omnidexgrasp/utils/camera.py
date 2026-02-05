"""📷 Camera utilities for intrinsic parameter computation."""
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """📐 Camera intrinsic parameters."""

    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int

#FIXME: define as class method
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
