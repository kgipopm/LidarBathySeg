from __future__ import annotations

import os
from typing import Optional, Tuple

import laspy
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# I/O helpers (light – full LAS export lives in io.py)
# ---------------------------------------------------------------------------

def load_pointcloud(file_path: str) -> np.ndarray:
    """Load a LAS file and return an (N, 5) array [x, y, z, classification, ID]."""
    las = laspy.read(file_path)
    ids = np.arange(las.z.shape[0], dtype=np.int64)
    return np.column_stack((las.x, las.y, las.z, las.classification, ids))


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def filter_pointcloud(
    raw: np.ndarray,
    z_low_pct: float = 1.0,
    z_high_pct: float = 99.0,
) -> np.ndarray:
    """Remove points outside the *z_low_pct*–*z_high_pct* Z-percentile range."""
    if raw.shape[1] < 3:
        raise ValueError("`raw` must have at least 3 columns (x, y, z).")
    zmin, zmax = np.percentile(raw[:, 2], (z_low_pct, z_high_pct))
    mask = (raw[:, 2] > zmin) & (raw[:, 2] < zmax)
    return raw[mask]


# ---------------------------------------------------------------------------
# Coordinate alignment
# ---------------------------------------------------------------------------

def _rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotate_to_principal_axis(
    cloud: np.ndarray,
    plot: bool = False,
) -> Tuple[np.ndarray, float]:
    """Rotate the XY plane so the dominant elongation axis aligns with X.

    Uses a linear fit of Y ~ X on a random subsample to estimate the
    principal direction. Returns (rotated_cloud, slope).
    """
    if cloud.shape[1] < 5:
        raise ValueError("`cloud` must have at least 5 columns [x, y, z, class, ID].")
    if cloud.shape[0] == 0:
        raise ValueError("`cloud` is empty.")

    n = cloud.shape[0]
    sample_size = min(n, max(300, n // 100))
    rng = default_rng(seed=42)
    idx = rng.choice(n, size=sample_size, replace=False)
    sub = cloud[idx]

    slope, intercept = np.polyfit(sub[:, 0], sub[:, 1], 1)
    angle = -np.arctan(slope)
    R = _rotation_matrix(angle)

    rotated = cloud.copy()
    rotated[:, :2] = cloud[:, :2] @ R.T

    if plot:
        x_line = np.linspace(sub[:, 0].min(), sub[:, 0].max(), 50)
        plt.figure()
        plt.plot(sub[:, 0], sub[:, 1], "o", ms=2, label="Sample")
        plt.plot(x_line, slope * x_line + intercept, "r-", label="Fit")
        plt.title("Original XY")
        plt.legend()
        plt.show()

        rot_sub = sub[:, :2] @ R.T
        plt.figure()
        plt.plot(rot_sub[:, 0], rot_sub[:, 1], "o", ms=2, label="Rotated")
        plt.title("Rotated XY")
        plt.legend()
        plt.show()

    return rotated, slope


def remove_bias(cloud: np.ndarray) -> np.ndarray:
    """Shift x, y, z so that their minima are 0. Modifies the array in-place."""
    if cloud.shape[1] < 3:
        raise ValueError("`cloud` must have at least 3 columns (x, y, z).")
    cloud[:, :3] -= cloud[:, :3].min(axis=0)
    return cloud


# ---------------------------------------------------------------------------
# Data augmentation (used by SVM training)
# ---------------------------------------------------------------------------

def extend_xy_with_nearest_z(
    points: np.ndarray,
    factor: float = 0.1,
    density_ratio: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic points in an extended XY bounding box.

    Z values are interpolated via nearest-neighbour from *points*.
    Returns only the *new* points (shape (N_new, 3)).
    """
    if points.shape[1] != 3:
        raise ValueError("`points` must have shape (N, 3).")
    if factor <= 0 or density_ratio <= 0:
        return np.empty((0, 3), dtype=points.dtype)

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    min_x, max_x = X.min(), X.max()
    min_y, max_y = Y.min(), Y.max()

    dx = (max_x - min_x) * factor / 2.0
    dy = (max_y - min_y) * factor / 2.0
    ex_min_x, ex_max_x = min_x - dx, max_x + dx
    ex_min_y, ex_max_y = min_y - dy, max_y + dy

    area_orig = (max_x - min_x) * (max_y - min_y)
    area_ext = (ex_max_x - ex_min_x) * (ex_max_y - ex_min_y)
    if area_orig <= 0:
        return np.empty((0, 3), dtype=points.dtype)

    n_new = int(points.shape[0] * (area_ext / area_orig - 1.0) * density_ratio)
    if n_new <= 0:
        return np.empty((0, 3), dtype=points.dtype)

    rng = default_rng(seed)
    new_X = rng.uniform(ex_min_x, ex_max_x, n_new)
    new_Y = rng.uniform(ex_min_y, ex_max_y, n_new)

    inside = (
        (new_X >= min_x) & (new_X <= max_x) &
        (new_Y >= min_y) & (new_Y <= max_y)
    )
    new_X, new_Y = new_X[~inside], new_Y[~inside]
    if new_X.size == 0:
        return np.empty((0, 3), dtype=points.dtype)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(points[:, :2])
    _, idx = nbrs.kneighbors(np.column_stack([new_X, new_Y]))
    new_Z = Z[idx[:, 0]]

    return np.column_stack([new_X, new_Y, new_Z])


def sample_xy(
    X: np.ndarray,
    y: np.ndarray,
    n: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly sample *n* matching rows from *X* and *y*."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if n > X.shape[0]:
        raise ValueError(f"Requested n={n} but only {X.shape[0]} available.")
    rng = default_rng(seed)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Quick diagnostics
# ---------------------------------------------------------------------------

def temporary_plot(pc: np.ndarray) -> None:
    """Display a 5 % subsample of the cloud in the XZ plane, coloured by class."""
    if pc.shape[1] < 4:
        raise ValueError("`pc` must have at least 4 columns [x, y, z, class].")
    sample_size = max(1, int(0.05 * pc.shape[0]))
    idx = default_rng().choice(pc.shape[0], sample_size, replace=False)
    sub = pc[idx]
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(sub[:, 0], sub[:, 2], c=sub[:, 3], cmap="viridis", alpha=0.6, s=1)
    plt.colorbar(sc, label="Class")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("XZ view (5 % sample)")
    plt.tight_layout()
    plt.show()
