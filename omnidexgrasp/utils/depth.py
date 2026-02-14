"""🌊 Depth rescaling utilities for aligning relative to metric depth."""
import numpy as np


def find_mask_center_coords(mask: np.ndarray, window_size: int = 20) -> np.ndarray:
    """🎯 Extract pixel coordinates in square window around mask centroid.

    Args:
        mask: Binary mask (H, W) uint8, non-zero pixels indicate object
        window_size: Square window side length (pixels)

    Returns:
        Pixel coords (N, 2) array of [row, col] within window, clipped to image bounds
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise ValueError("❌ Mask is empty, cannot find centroid")

    center_r = int(np.median(ys))
    center_c = int(np.median(xs))

    half = window_size // 2
    h, w = mask.shape
    coords = []

    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r = center_r + dr
            c = center_c + dc
            if 0 <= r < h and 0 <= c < w:
                coords.append([r, c])

    return np.array(coords)


def compute_depth_affine(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    pixel_coords: np.ndarray,
    mask_invalid: bool = True,
) -> tuple[float, float]:
    """📐 Compute affine transformation: alpha * depth_pred + beta ≈ depth_gt.

    Uses least squares to solve for alpha (scale) and beta (shift) using
    only pixels specified by pixel_coords.

    Args:
        depth_pred: Predicted depth (H, W) float32
        depth_gt: Ground truth depth (H, W) float32
        pixel_coords: Pixel coords (N, 2) array of [row, col]
        mask_invalid: If True, ignore pixels where either depth is zero

    Returns:
        (alpha, beta) where transformed depth = alpha * depth_pred + beta
    """
    rows = pixel_coords[:, 0]
    cols = pixel_coords[:, 1]

    dp_vals = depth_pred[rows, cols].astype(np.float32)
    dg_vals = depth_gt[rows, cols].astype(np.float32)

    if mask_invalid:
        valid = (dp_vals > 0) & (dg_vals > 0)
        if not np.any(valid):
            raise ValueError("❌ No valid depth pairs found after masking")
        dp_vals = dp_vals[valid]
        dg_vals = dg_vals[valid]

    # Solve [dp_vals, 1] * [alpha; beta] = dg_vals
    A = np.stack([dp_vals, np.ones_like(dp_vals)], axis=-1)
    coefs, *_ = np.linalg.lstsq(A, dg_vals, rcond=None)
    alpha = float(coefs[0])
    beta = float(coefs[1])

    return alpha, beta


def rescale_depth(
    depth: np.ndarray, alpha: float, beta: float, clip_negative: bool = True
) -> np.ndarray:
    """🔄 Apply affine transformation to depth: y = alpha * x + beta.

    Args:
        depth: Input depth (H, W) float32
        alpha: Scale factor
        beta: Shift offset
        clip_negative: If True, set negative values to 0

    Returns:
        Rescaled depth (H, W) float32
    """
    rescaled = alpha * depth + beta
    if clip_negative:
        rescaled = np.maximum(rescaled, 0.0)
    return rescaled.astype(np.float32)
