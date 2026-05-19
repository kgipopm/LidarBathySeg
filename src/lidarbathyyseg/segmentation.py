from __future__ import annotations

import os
import re
from typing import Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from .visualization import build_html_with_axis_controls


def filter_points(
    points_np: np.ndarray,
    stable_profile: dict,
    output_dir: Optional[str] = None,
    save_html: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separate points into *seabed* and *water* using the stable Z-band.

    Decision logic
    --------------
    * ``position == "above"``: seabed is *above* the band →
      keep points with Z < ``range_min`` as seabed, rest as water.
    * ``position == "below"``: seabed is *below* the band →
      keep points with Z > ``range_max`` as seabed, rest as water.
    * ``position == "middle"``: seabed *is* the band.

    Parameters
    ----------
    points_np:
        (N, 5) array [X, Y, Z, class, ID].
    stable_profile:
        Output of :func:`get_universal_stable_profile`.
    output_dir:
        Directory to write the interactive HTML plot. ``None`` disables saving.
    save_html:
        Set ``False`` to skip HTML generation entirely (useful during testing).

    Returns
    -------
    seabed_xyz, water_xyz, seabed_ids, water_ids
        Full-resolution ID arrays; XYZ arrays are downsampled to ≤ 1000 pts
        for the visualisation only (IDs are always complete).
    """
    if points_np.ndim != 2 or points_np.shape[1] < 5:
        raise ValueError("points_np must be (N, 5): [X, Y, Z, class, ID].")

    z = points_np[:, 2]
    zmin_band = stable_profile["range_min"]
    zmax_band = stable_profile["range_max"]
    position = stable_profile.get("position", "middle")

    if position == "above":
        mask_seabed = z < zmin_band
    elif position == "below":
        mask_seabed = z > zmax_band
    else:
        mask_seabed = (z >= zmin_band) & (z <= zmax_band)

    seabed_raw = points_np[mask_seabed]
    water_raw = points_np[~mask_seabed]

    rng = np.random.default_rng()

    def _ds(arr: np.ndarray, n: int = 1_000) -> np.ndarray:
        if arr.shape[0] <= n:
            return arr
        return arr[rng.choice(arr.shape[0], n, replace=False)]

    seabed_vis = _ds(seabed_raw)
    water_vis = _ds(water_raw)

    if save_html and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        fig = go.Figure()
        if seabed_vis.shape[0]:
            fig.add_trace(go.Scatter3d(
                x=seabed_vis[:, 0], y=seabed_vis[:, 1], z=seabed_vis[:, 2],
                mode="markers", marker=dict(size=2, color="blue"),
                name="Seabed (remaining)",
            ))
        if water_vis.shape[0]:
            fig.add_trace(go.Scatter3d(
                x=water_vis[:, 0], y=water_vis[:, 1], z=water_vis[:, 2],
                mode="markers", marker=dict(size=2, color="red", opacity=0.5),
                name="Water (removed)",
            ))

        all_pts = np.vstack([seabed_vis[:, :3], water_vis[:, :3]]) if seabed_vis.shape[0] + water_vis.shape[0] else np.zeros((1, 3))
        lo = np.min(all_pts, axis=0)
        hi = np.max(all_pts, axis=0)
        mid = (lo + hi) / 2
        half = max(hi - lo) / 2 or 1.0

        xr = [mid[0] - half, mid[0] + half]
        yr = [mid[1] - half, mid[1] + half]
        zr = [mid[2] - half, mid[2] + half]

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X", range=xr),
                yaxis=dict(title="Y", range=yr),
                zaxis=dict(title="Z", range=zr),
                aspectmode="cube",
            ),
            title=f"Separation ({position})",
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
        )

        title = f"Separation_relative_to_stable_band_{position}"
        html = build_html_with_axis_controls(
            title=title,
            inner_html=fig.to_html(include_plotlyjs="cdn", full_html=False),
            xrange_init=xr, yrange_init=yr, zrange_init=zr,
        )
        safe = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
        with open(os.path.join(output_dir, f"{safe}.html"), "w", encoding="utf-8") as fh:
            fh.write(html)

    return (
        seabed_vis[:, :3],
        water_vis[:, :3],
        seabed_raw[:, 4],
        water_raw[:, 4],
    )


# Backward-compatible alias
filter_and_visualize_numpy_points = filter_points


def assign_classes(
    pointcloud_raw: np.ndarray,
    seabed_ids: Sequence[int],
    water_ids: Sequence[int],
) -> np.ndarray:
    """Assign class labels by point ID.

    Class map: 1 = seabed, 0 = water, 30 = unclassified (default).
    """
    if pointcloud_raw.ndim != 2 or pointcloud_raw.shape[1] < 5:
        raise ValueError("pointcloud_raw must be (N, 5): [X, Y, Z, class, ID].")

    out = pointcloud_raw.copy()
    ids = out[:, 4].astype(int)
    classes = np.full(ids.shape, 30, dtype=out.dtype)
    classes[np.isin(ids, seabed_ids)] = 1
    classes[np.isin(ids, water_ids)] = 0
    out[:, 3] = classes
    return out
