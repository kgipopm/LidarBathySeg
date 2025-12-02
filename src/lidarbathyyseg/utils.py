from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import laspy
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn.neighbors import NearestNeighbors


def sample_xy(
    X: np.ndarray,
    y: np.ndarray,
    n: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample n elements from X and y using the same indices.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N, d).
    y : np.ndarray
        Target array of shape (N,) or (N, k).
    n : int, optional
        Number of samples to draw (default 1000).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_sample : np.ndarray
        Randomly selected rows from X.
    y_sample : np.ndarray
        Corresponding rows from y.

    Raises
    ------
    ValueError
        If n is greater than the number of available samples or if
        X and y have inconsistent lengths.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    if n > X.shape[0]:
        raise ValueError(
            f"Requested n={n} samples, but only {X.shape[0]} available."
        )

    rng = default_rng(seed)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]


def extend_xy_with_nearest_z(
    points: np.ndarray,
    factor: float = 0.1,
    density_ratio: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate new points in an extended XY bounding box and interpolate Z
    using nearest-neighbor interpolation.

    Parameters
    ----------
    points : np.ndarray
        Array (N, 3) with columns [X, Y, Z].
    factor : float, optional
        Percentage by which to extend the bounding box in XY
        (e.g. 0.1 = +10%). Default is 0.1.
    density_ratio : float, optional
        Controls density of new points relative to original.
        1.0 ≈ similar spacing in the extended area. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    new_points : np.ndarray
        Generated points of shape (N_new, 3) with columns [X, Y, Z].

    Notes
    -----
    - This function returns only the newly generated points, not the
      original points plus the new ones.
    - If factor <= 0 or the computed number of new points is <= 0,
      an empty array is returned.
    """
    if points.shape[1] != 3:
        raise ValueError("`points` must have shape (N, 3) with columns [X, Y, Z].")

    if factor <= 0 or density_ratio <= 0:
        return np.empty((0, 3), dtype=points.dtype)

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    min_x, max_x = X.min(), X.max()
    min_y, max_y = Y.min(), Y.max()

    dx = (max_x - min_x) * factor / 2.0
    dy = (max_y - min_y) * factor / 2.0
    new_min_x, new_max_x = min_x - dx, max_x + dx
    new_min_y, new_max_y = min_y - dy, max_y + dy

    area_orig = (max_x - min_x) * (max_y - min_y)
    area_new = (new_max_x - new_min_x) * (new_max_y - new_min_y)

    if area_orig <= 0:
        return np.empty((0, 3), dtype=points.dtype)

    n_new = int(points.shape[0] * (area_new / area_orig - 1.0) * density_ratio)
    if n_new <= 0:
        return np.empty((0, 3), dtype=points.dtype)

    rng = default_rng(seed)
    new_X = rng.uniform(new_min_x, new_max_x, n_new)
    new_Y = rng.uniform(new_min_y, new_max_y, n_new)

    mask_inside = (
        (new_X >= min_x) & (new_X <= max_x) &
        (new_Y >= min_y) & (new_Y <= max_y)
    )
    new_X, new_Y = new_X[~mask_inside], new_Y[~mask_inside]

    if new_X.size == 0:
        return np.empty((0, 3), dtype=points.dtype)

    new_xy = np.column_stack([new_X, new_Y])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(points[:, :2])
    _, idx = nbrs.kneighbors(new_xy)
    new_Z = Z[idx[:, 0]]

    new_points = np.column_stack([new_X, new_Y, new_Z])
    return new_points


def load_pointcloud(file_path: str) -> np.ndarray:
    """
    Load a point cloud from a LAS file and return it as a NumPy array.

    Parameters
    ----------
    file_path : str
        Path to the LAS file to be processed.

    Returns
    -------
    np.ndarray
        Array of shape (N, 5) with columns:
        [x, y, z, classification, ID].

    Notes
    -----
    - The `ID` column is assigned sequentially from 0 to N-1.
    - Requires the `laspy` library for reading LAS files.
    """
    las_file = laspy.read(file_path)

    X = las_file.x
    Y = las_file.y
    Z = las_file.z
    cls_ = las_file.classification
    ids = np.arange(Z.shape[0], dtype=np.int64)

    raw_pointcloud = np.column_stack((X, Y, Z, cls_, ids))
    return raw_pointcloud


def filter_pointcloud(raw_pointcloud: np.ndarray) -> np.ndarray:
    """
    Filter a point cloud by clipping extreme Z values.

    Currently, this removes points outside the 1st and 99th percentiles
    of the Z distribution.

    Parameters
    ----------
    raw_pointcloud : np.ndarray
        Input data with columns at least [x, y, z, ...].

    Returns
    -------
    filtered_pointcloud : np.ndarray
        Subset of the input with Z in (1st, 99th) percentile range.
    """
    if raw_pointcloud.shape[1] < 3:
        raise ValueError("`raw_pointcloud` must have at least 3 columns (x, y, z).")

    zmin, zmax = np.percentile(raw_pointcloud[:, 2], (1, 99))
    mask = (raw_pointcloud[:, 2] > zmin) & (raw_pointcloud[:, 2] < zmax)
    filtered_pointcloud = raw_pointcloud[mask]
    return filtered_pointcloud


def _rotation_matrix(angle: float) -> np.ndarray:
    """
    Generate a 2D rotation matrix for a given angle in radians.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        A 2x2 rotation matrix:

            [[cos(angle), -sin(angle)],
             [sin(angle),  cos(angle)]]
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotate_to_principal_axis(
    filtered_pointcloud: np.ndarray,
    plot: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Rotate a point cloud to align its XY plane with an estimated principal direction.

    The principal direction is approximated by a linear fit of Y on X over a random
    subset of points. The slope of this fit defines a rotation angle that is applied
    to all XY coordinates, leaving Z, classification and ID unchanged.

    Parameters
    ----------
    filtered_pointcloud : np.ndarray
        Array of shape (N, 5) with columns:
        [x, y, z, classification, ID].
    plot : bool, optional
        If True, shows diagnostic matplotlib plots before and after rotation.
        Default is False.

    Returns
    -------
    rotated_pointcloud : np.ndarray
        Array of shape (N, 5) with rotated x and y.
    rotation_slope : float
        Slope of the fitted line (Y ~ X). The angle used is
        angle = -arctan(rotation_slope).

    Notes
    -----
    - A deterministic random subset of max(N // 100, 300) points is drawn
      (but not exceeding N) using numpy.random.default_rng(seed=42).
    """
    if filtered_pointcloud.shape[1] < 5:
        raise ValueError(
            "`filtered_pointcloud` must have at least 5 columns "
            "[x, y, z, classification, ID]."
        )

    n_points = filtered_pointcloud.shape[0]
    if n_points == 0:
        raise ValueError("`filtered_pointcloud` is empty.")

    sample_target = max(300, n_points // 100)
    sample_size = min(n_points, sample_target)

    rng = default_rng(seed=42)
    idx = rng.choice(n_points, size=sample_size, replace=False)
    pointcloud_sampled = filtered_pointcloud[idx, :]

    X = pointcloud_sampled[:, 0]
    Y = pointcloud_sampled[:, 1]

    rotation_slope, rotation_intercept = np.polyfit(X, Y, 1)
    rotation_angle = -np.arctan(rotation_slope)

    coords_all = filtered_pointcloud[:, :2]  # [x, y]
    R = _rotation_matrix(rotation_angle)
    new_coords_all = coords_all @ R.T

    rotated_pointcloud = filtered_pointcloud.copy()
    rotated_pointcloud[:, 0] = new_coords_all[:, 0]
    rotated_pointcloud[:, 1] = new_coords_all[:, 1]

    if plot:
        x_line = np.linspace(X.min(), X.max(), 50)
        y_line = rotation_slope * x_line + rotation_intercept

        plt.figure()
        plt.plot(X, Y, "o", ms=2, label="Sampled points")
        plt.plot(x_line, y_line, "r-", label="Fitted line")
        plt.legend()
        plt.title("Original (XY)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

        coords_line = np.column_stack((x_line, y_line))
        coords_sample = np.column_stack((X, Y))
        new_line = coords_line @ R.T
        new_sample = coords_sample @ R.T

        plt.figure()
        plt.plot(new_sample[:, 0], new_sample[:, 1], "o", ms=2, label="Rotated sample")
        plt.plot(new_line[:, 0], new_line[:, 1], "r-", label="Rotated line")
        plt.legend()
        plt.title("Rotated (XY)")
        plt.xlabel("X'")
        plt.ylabel("Y'")
        plt.show()

    return rotated_pointcloud, rotation_slope


def remove_bias(point_cloud: np.ndarray) -> np.ndarray:
    """
    Shift a point cloud so that the minimum of each coordinate axis is zero.

    Parameters
    ----------
    point_cloud : np.ndarray
        Array of shape (N, 5) with columns:
        [x, y, z, classification, ID].

    Returns
    -------
    adjusted_cloud : np.ndarray
        The same array with x, y, z shifted so that their minima are 0.

    Notes
    -----
    - Operation is in-place: the input array is modified and also returned.
    """
    if point_cloud.shape[1] < 3:
        raise ValueError("`point_cloud` must have at least 3 columns (x, y, z).")

    point_cloud[:, :3] -= point_cloud[:, :3].min(axis=0)
    return point_cloud


def chunk_pointcloud(
    unbiased_pointcloud: np.ndarray,
    x_bins: int = 5,
) -> List[Dict[str, np.ndarray | float | Tuple[int, ...]]]:
    """
    Split a point cloud into equally sized chunks along the X-axis.

    Parameters
    ----------
    unbiased_pointcloud : np.ndarray
        Array of shape (N, M), with at least x in column 0.
    x_bins : int, optional
        Number of chunks to create along the X-axis (default is 5).

    Returns
    -------
    list of dict
        Each dict has keys:
        - 'chunk' (np.ndarray)
        - 'upper' (float)
        - 'lower' (float)
        - 'shape' (tuple)
    """
    if unbiased_pointcloud.shape[1] == 0:
        raise ValueError("`unbiased_pointcloud` must have at least 1 column.")

    x_max = unbiased_pointcloud[:, 0].max()
    if x_bins <= 0:
        raise ValueError("`x_bins` must be a positive integer.")

    dx = x_max / x_bins if x_bins > 0 else x_max
    chunks: List[Dict[str, np.ndarray | float | Tuple[int, ...]]] = []

    for i in range(x_bins):
        lower = i * dx
        upper = (i + 1) * dx if i < x_bins - 1 else x_max

        if i < x_bins - 1:
            mask = (unbiased_pointcloud[:, 0] >= lower) & (unbiased_pointcloud[:, 0] < upper)
        else:
            mask = (unbiased_pointcloud[:, 0] >= lower) & (unbiased_pointcloud[:, 0] <= upper)

        chunk = unbiased_pointcloud[mask]
        chunks.append(
            {
                "chunk": chunk,
                "upper": float(upper),
                "lower": float(lower),
                "shape": chunk.shape,
            }
        )

    return chunks


def temporary_plot(pc: np.ndarray) -> None:
    """
    Visualize a sampled subset of a point cloud in the XZ plane.

    Parameters
    ----------
    pc : np.ndarray
        Array of shape (N, M) with at least columns:
        [x, y, z, classification, ...].

    Returns
    -------
    None
    """
    if pc.shape[1] < 4:
        raise ValueError(
            "`pc` must have at least 4 columns: [x, y, z, classification]."
        )

    sample_size = max(1, int(0.05 * pc.shape[0]))
    rng = default_rng()
    sample_indices = rng.choice(pc.shape[0], sample_size, replace=False)
    sampled_data = pc[sample_indices]

    x_values = sampled_data[:, 0]
    z_values = sampled_data[:, 2]
    classes = sampled_data[:, 3]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x_values, z_values,
        c=classes, cmap="viridis", alpha=0.6, s=1,
    )
    plt.colorbar(scatter, label="Object class")
    plt.xlabel("X - Horizontal coordinate")
    plt.ylabel("Z - Height")
    plt.title("Visualization of object classes in XZ plane")
    plt.tight_layout()
    plt.show()


def save_results_to_original_pointcloud(
    input_las: str,
    concatenated_id_class: np.ndarray,
    output_las: Optional[str] = None,
    default_class: int = 20,
) -> None:
    """
    Save updated classifications into a LAS point cloud file.

    Parameters
    ----------
    input_las : str
        Path to the input LAS file.
    concatenated_id_class : np.ndarray
        Array of shape (N, 2):
        - column 0: integer point indices
        - column 1: new classification values (uint8).
    output_las : str, optional
        Output LAS path. If None, 'modified_<input_filename>.las'
        will be used.
    default_class : int, optional
        Default classification for points not listed in
        `concatenated_id_class` (default 20).

    Returns
    -------
    None
    """
    if concatenated_id_class.ndim != 2 or concatenated_id_class.shape[1] != 2:
        raise ValueError(
            "`concatenated_id_class` must have shape (N, 2): [index, class]."
        )

    if output_las is None:
        filename = os.path.basename(input_las)
        output_las = f"modified_{filename}"

    with laspy.open(input_las, mode="r") as las_file:
        las = las_file.read()

    num_points = len(las.classification)
    new_classifications = np.full(num_points, default_class, dtype=np.uint8)

    indices = concatenated_id_class[:, 0].astype(int)
    values = concatenated_id_class[:, 1].astype(np.uint8)

    valid_mask = (indices >= 0) & (indices < num_points)
    new_classifications[indices[valid_mask]] = values[valid_mask]

    las.classification = new_classifications
    las.write(output_las)
