from __future__ import annotations

import os
from typing import Optional

import laspy
import numpy as np


def save_to_las(updated_array: np.ndarray, output_dir: str) -> str:
    """Write a (N, 5) point-cloud array [x, y, z, class, ID] to a LAS file.

    Returns the path to the saved file.
    """
    if updated_array.ndim != 2 or updated_array.shape[1] < 5:
        raise ValueError("updated_array must have shape (N, 5): [x, y, z, class, ID].")

    os.makedirs(output_dir, exist_ok=True)

    x = updated_array[:, 0]
    y = updated_array[:, 1]
    z = updated_array[:, 2]
    classification = updated_array[:, 3].astype(np.uint8)
    point_ids = updated_array[:, 4].astype(np.uint64)

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.array([x.min(), y.min(), z.min()])
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.classification = classification
    las.user_data = point_ids

    filepath = os.path.join(output_dir, "updated_pointcloud.las")
    las.write(filepath)
    print(f"Saved: {filepath}")
    return filepath


def save_results_to_original_las(
    input_las: str,
    concatenated_id_class: np.ndarray,
    output_las: Optional[str] = None,
    default_class: int = 20,
) -> str:
    """Patch an existing LAS file with new per-point classification values.

    Parameters
    ----------
    input_las:
        Path to the source LAS file.
    concatenated_id_class:
        (N, 2) array where column 0 = point index, column 1 = new class.
    output_las:
        Destination path. Defaults to ``modified_<basename>``.
    default_class:
        Class assigned to every point not listed in *concatenated_id_class*.

    Returns the path to the written file.
    """
    if concatenated_id_class.ndim != 2 or concatenated_id_class.shape[1] != 2:
        raise ValueError("`concatenated_id_class` must have shape (N, 2).")

    if output_las is None:
        output_las = f"modified_{os.path.basename(input_las)}"

    with laspy.open(input_las, mode="r") as f:
        las = f.read()

    n = len(las.classification)
    new_cls = np.full(n, default_class, dtype=np.uint8)

    indices = concatenated_id_class[:, 0].astype(int)
    values = concatenated_id_class[:, 1].astype(np.uint8)
    valid = (indices >= 0) & (indices < n)
    new_cls[indices[valid]] = values[valid]

    las.classification = new_cls
    las.write(output_las)
    print(f"Saved: {output_las}")
    return output_las
