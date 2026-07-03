from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from joblib import Parallel, delayed
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

from .visualization import build_html_with_axis_controls


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _smooth_row(
    i: int,
    cx_arr: np.ndarray,
    cy_arr: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    rx: float,
    ry: float,
    sigma: float,
) -> Tuple[int, np.ndarray]:
    """Compute one grid row of the smoothed surface (parallelisable unit)."""
    grid_cols = len(cx_arr)
    Z_row = np.full(grid_cols, np.nan)

    for j in range(grid_cols):
        cx, cy = cx_arr[j], cy_arr[j]
        dxs = (X - cx) / rx
        dys = (Y - cy) / ry
        inside = (dxs ** 2 + dys ** 2) <= 1.0

        if not np.any(inside):
            continue

        lZ = Z[inside]
        p10, p90 = np.percentile(lZ, [10, 90])
        valid = (lZ >= p10) & (lZ <= p90)
        if valid.sum() < 3:
            continue

        lZ = lZ[valid]
        lXY = np.stack([X[inside][valid], Y[inside][valid]], axis=1)

        dists = cdist([[cx, cy]], lXY)[0]
        w = np.exp(-(dists ** 2) / (2 * sigma ** 2))
        s = w.sum()
        if s == 0:
            continue

        Z_row[j] = np.dot(w / s, lZ)

    return i, Z_row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smooth_surface_from_points(
    points_np: np.ndarray,
    grid_size: int = 30,
    alpha: float = 0.55,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a smooth XY surface grid from scattered seabed points.

    Algorithm
    ---------
    For each grid node the algorithm:

    1. Collects data points inside an elliptical neighbourhood of radius
       ``alpha × cell_size`` along each axis.
    2. Removes outliers (10th–90th percentile of local Z values).
    3. Computes a Gaussian-weighted mean Z (bandwidth = 0.5 × max radius).
    4. Fills remaining NaN cells via ``scipy.griddata`` cubic interpolation.

    The outer loop over grid rows is parallelised with *joblib* using threads
    (no pickling overhead because numpy releases the GIL).

    Parameters
    ----------
    points_np:
        (N, ≥3) array [X, Y, Z, ...].
    grid_size:
        Number of nodes along each axis (grid_size × grid_size total).
    alpha:
        Ellipse-radius multiplier relative to the grid cell size.
    n_jobs:
        Passed to :class:`joblib.Parallel`. ``-1`` = all CPU cores.

    Returns
    -------
    X_grid, Y_grid, Z_interp_grid : np.ndarray
        Three (grid_size, grid_size) arrays.
    """
    if points_np.ndim != 2 or points_np.shape[1] < 3:
        raise ValueError("points_np must be (N, ≥3).")

    X, Y, Z = points_np[:, 0], points_np[:, 1], points_np[:, 2]

    x_lin = np.linspace(X.min(), X.max(), grid_size)
    y_lin = np.linspace(Y.min(), Y.max(), grid_size)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)

    dx = np.diff(x_lin).mean()
    dy = np.diff(y_lin).mean()
    rx = alpha * dx
    ry = alpha * dy
    sigma = 0.5 * max(rx, ry)

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_smooth_row)(i, X_grid[i], Y_grid[i], X, Y, Z, rx, ry, sigma)
        for i in range(grid_size)
    )

    Z_grid = np.full_like(X_grid, np.nan, dtype=float)
    for i, row in rows:
        Z_grid[i] = row

    known_mask = ~np.isnan(Z_grid)
    known_pts = np.stack([X_grid[known_mask], Y_grid[known_mask]], axis=1)
    known_vals = Z_grid[known_mask]
    all_pts = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
    Z_interp = griddata(known_pts, known_vals, all_pts, method="cubic")
    Z_interp_grid = Z_interp.reshape(X_grid.shape)

    return X_grid, Y_grid, Z_interp_grid


def plot_surface(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z_grid: np.ndarray,
    output_dir: str,
) -> str:
    """Save an interactive 3-D surface plot as an HTML file.

    Returns the path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = go.Figure(data=[go.Surface(
        z=Z_grid, x=X_grid, y=Y_grid,
        colorscale="Viridis", opacity=0.9,
    )])
    fig.update_layout(
        title="Interpolated seabed surface",
        scene=dict(xaxis_title="X [m]", yaxis_title="Y [m]", zaxis_title="Z [m]"),
    )

    xr = (float(np.nanmin(X_grid)), float(np.nanmax(X_grid)))
    yr = (float(np.nanmin(Y_grid)), float(np.nanmax(Y_grid)))
    zr = (float(np.nanmin(Z_grid)), float(np.nanmax(Z_grid)))

    html = build_html_with_axis_controls(
        title="Interpolated seabed surface",
        inner_html=fig.to_html(full_html=False, include_plotlyjs="cdn"),
        xrange_init=xr, yrange_init=yr, zrange_init=zr,
    )

    filepath = os.path.join(output_dir, "seabed_surface.html")
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(html)
    return filepath
