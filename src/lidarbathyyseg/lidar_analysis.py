from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import laspy
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from joblib import Parallel, delayed
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .utils import extend_xy_with_nearest_z, sample_xy


def _build_html_with_axis_controls(
    title: str,
    inner_html: str,
    xrange_init: Sequence[float],
    yrange_init: Sequence[float],
    zrange_init: Sequence[float],
    note_text: str = "Każda zmiana przepisuje zakres danej osi przez Plotly.relayout().",
) -> str:
    """
    Zbuduj HTML z wykresem Plotly oraz panelem bocznym z kontrolą zakresów osi.
    """
    x0, x1 = xrange_init
    y0, y1 = yrange_init
    z0, z1 = zrange_init

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{
    font-family: sans-serif;
    margin: 0;
    display: grid;
    grid-template-columns: 1fr 260px;
    height: 100vh;
  }}

  #plot-wrapper {{
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    height: 100vh;
  }}

  #plot-container {{
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
    height: 100%;
  }}

  /* Dodatkowy wrapper, żeby mieć pełną kontrolę nad wysokością */
  #plot-inner {{
    height: 100%;
    width: 100%;
  }}

  /* Upewniamy się, że graf wypełnia cały wrapper, niezależnie od struktury wewnątrz */
  #plot-inner .js-plotly-plot,
  #plot-inner .plotly-graph-div {{
    height: 100% !important;
    width: 100% !important;
  }}

  #controls {{
    background: #f4f4f4;
    border-left: 1px solid #ccc;
    padding: 1rem;
    overflow-y: auto;
    font-size: 14px;
    line-height: 1.4;
  }}

  .ctrl-block {{
    margin-bottom: 1rem;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 0.75rem;
  }}

  .ctrl-block h3 {{
    margin: 0 0 0.5rem 0;
    font-size: 14px;
    font-weight: 600;
  }}

  .row {{
    display: flex;
    align-items: center;
    gap: .5rem;
    margin-bottom: .5rem;
  }}

  .row label {{
    width: 45px;
    font-weight: 600;
    font-size: 13px;
  }}

  .row input[type=range] {{
    flex: 1;
  }}

  .row input[type=number] {{
    width: 70px;
  }}

  button {{
    width: 100%;
    padding: .5rem .75rem;
    font-size: 14px;
    font-weight: 600;
    border-radius: 6px;
    border: 1px solid #888;
    background: #eee;
    cursor: pointer;
  }}

  button:hover {{
    background: #ddd;
  }}

  small {{
    color: #666;
    font-size: 12px;
  }}
</style>
</head>

<body>

<div id="plot-wrapper">
  <div id="plot-container">
    <div id="plot-inner">
      {inner_html}
    </div>
  </div>
</div>

<aside id="controls">

  <div class="ctrl-block">
    <h3>Reset view</h3>
    <button id="resetBtn">Reset to cube</button>
  </div>

  <div class="ctrl-block">
    <h3>Axis ranges</h3>

    <!-- X axis -->
    <div class="row">
      <label>X min</label>
      <input id="xMinNum" type="number" step="0.1" value="{x0:.2f}">
    </div>
    <div class="row">
      <label>X max</label>
      <input id="xMaxRange" type="range"
             min="{x0:.2f}"
             max="{x1*2:.2f}"
             step="0.1"
             value="{x1:.2f}">
      <input id="xMaxNum" type="number" step="0.1" value="{x1:.2f}">
    </div>

    <!-- Y axis -->
    <div class="row">
      <label>Y min</label>
      <input id="yMinNum" type="number" step="0.1" value="{y0:.2f}">
    </div>
    <div class="row">
      <label>Y max</label>
      <input id="yMaxRange" type="range"
             min="{y0:.2f}"
             max="{y1*2:.2f}"
             step="0.1"
             value="{y1:.2f}">
      <input id="yMaxNum" type="number" step="0.1" value="{y1:.2f}">
    </div>

    <!-- Z axis -->
    <div class="row">
      <label>Z min</label>
      <input id="zMinNum" type="number" step="0.1" value="{z0:.2f}">
    </div>
    <div class="row">
      <label>Z max</label>
      <input id="zMaxRange" type="range"
             min="{z0:.2f}"
             max="{z1*2:.2f}"
             step="0.1"
             value="{z1:.2f}">
      <input id="zMaxNum" type="number" step="0.1" value="{z1:.2f}">
    </div>

    <small>{note_text}</small>
  </div>

</aside>

<script>
// teraz bierzemy wykres z #plot-inner, nie bezpośrednio z #plot-container
const plotDiv = document
  .querySelector('#plot-inner')
  .querySelector('div.js-plotly-plot');

const initRanges = {{
  x: [{x0}, {x1}],
  y: [{y0}, {y1}],
  z: [{z0}, {z1}]
}};

function setAxisRange(axis, minVal, maxVal) {{
  const relayoutUpdate = {{}};
  relayoutUpdate[`scene.${{axis}}axis.range`] = [minVal, maxVal];
  Plotly.relayout(plotDiv, relayoutUpdate);
}}

const xMinNum = document.getElementById('xMinNum');
const xMaxRange = document.getElementById('xMaxRange');
const xMaxNum = document.getElementById('xMaxNum');

function updateXFromInputs() {{
  const xmin = parseFloat(xMinNum.value);
  const xmax = parseFloat(xMaxNum.value);
  if (!isNaN(xmin) && !isNaN(xmax) && xmax > xmin) {{
    setAxisRange('x', xmin, xmax);
  }}
}}

xMinNum.addEventListener('change', updateXFromInputs);
xMaxRange.addEventListener('input', (e) => {{
  xMaxNum.value = e.target.value;
  updateXFromInputs();
}});
xMaxNum.addEventListener('change', () => {{
  xMaxRange.value = xMaxNum.value;
  updateXFromInputs();
}});

const yMinNum = document.getElementById('yMinNum');
const yMaxRange = document.getElementById('yMaxRange');
const yMaxNum = document.getElementById('yMaxNum');

function updateYFromInputs() {{
  const ymin = parseFloat(yMinNum.value);
  const ymax = parseFloat(yMaxNum.value);
  if (!isNaN(ymin) && !isNaN(ymax) && ymax > ymin) {{
    setAxisRange('y', ymin, ymax);
  }}
}}

yMinNum.addEventListener('change', updateYFromInputs);
yMaxRange.addEventListener('input', (e) => {{
  yMaxNum.value = e.target.value;
  updateYFromInputs();
}});
yMaxNum.addEventListener('change', () => {{
  yMaxRange.value = yMaxNum.value;
  updateYFromInputs();
}});

const zMinNum = document.getElementById('zMinNum');
const zMaxRange = document.getElementById('zMaxRange');
const zMaxNum = document.getElementById('zMaxNum');

function updateZFromInputs() {{
  const zmin = parseFloat(zMinNum.value);
  const zmax = parseFloat(zMaxNum.value);
  if (!isNaN(zmin) && !isNaN(zmax) && zmax > zmin) {{
    setAxisRange('z', zmin, zmax);
  }}
}}

zMinNum.addEventListener('change', updateZFromInputs);
zMaxRange.addEventListener('input', (e) => {{
  zMaxNum.value = e.target.value;
  updateZFromInputs();
}});
zMaxNum.addEventListener('change', () => {{
  zMaxRange.value = zMaxNum.value;
  updateZFromInputs();
}});

document.getElementById('resetBtn').addEventListener('click', () => {{
  xMinNum.value = initRanges.x[0];
  xMaxNum.value = initRanges.x[1];
  xMaxRange.value = initRanges.x[1];

  yMinNum.value = initRanges.y[0];
  yMaxNum.value = initRanges.y[1];
  yMaxRange.value = initRanges.y[1];

  zMinNum.value = initRanges.z[0];
  zMaxNum.value = initRanges.z[1];
  zMaxRange.value = initRanges.z[1];

  Plotly.relayout(plotDiv, {{
    'scene.xaxis.range': initRanges.x,
    'scene.yaxis.range': initRanges.y,
    'scene.zaxis.range': initRanges.z,
    'scene.aspectmode': 'cube'
  }});
}});
</script>

</body>
</html>
"""


def check_best_gmm_model(
    points_xyz: np.ndarray,
    path: Optional[str] = None,
    saveplot: bool = True,
) -> Dict[str, Any]:
    """
    Fit 1- and 2-component Gaussian Mixture Models (GMM) to Z-values and select
    the better model via BIC.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError(
            "points_xyz must have shape (N, 3) or more; got "
            f"{points_xyz.shape}"
        )

    if path is None:
        path = os.getcwd()
    os.makedirs(path, exist_ok=True)

    z = points_xyz[:, 2].reshape(-1, 1)

    bics: List[float] = []
    aics: List[float] = []
    models: List[GaussianMixture] = []
    components_list = [1, 2]
    results: Dict[str, Any] = {}

    for n_components in components_list:
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(z)
        bics.append(float(gmm.bic(z)))
        aics.append(float(gmm.aic(z)))
        models.append(gmm)

    best_model_idx = int(np.argmin(bics))
    best_components = components_list[best_model_idx]

    gmm_1 = models[0]
    mean_1 = float(gmm_1.means_[0, 0])
    std_1 = float(np.sqrt(gmm_1.covariances_[0, 0]))
    amp_1 = float(gmm_1.weights_[0])

    results["1_component"] = {
        "mean": mean_1,
        "std": std_1,
        "amplitude": amp_1,
    }

    gmm_2 = models[1]
    means_2 = gmm_2.means_.flatten()
    stds_2 = np.sqrt(gmm_2.covariances_).flatten()
    amps_2 = gmm_2.weights_.flatten()

    sorted_indices = np.argsort(-means_2)  # descending
    means_sorted = means_2[sorted_indices]
    stds_sorted = stds_2[sorted_indices]
    amps_sorted = amps_2[sorted_indices]

    results["2_components"] = [
        {
            "mean": float(means_sorted[0]),
            "std": float(stds_sorted[0]),
            "amplitude": float(amps_sorted[0]),
        },
        {
            "mean": float(means_sorted[1]),
            "std": float(stds_sorted[1]),
            "amplitude": float(amps_sorted[1]),
        },
    ]

    results["bic"] = {"1": bics[0], "2": bics[1]}
    results["aic"] = {"1": aics[0], "2": aics[1]}
    results["best_model"] = best_components

    if saveplot:
        z_lin = np.linspace(z.min(), z.max(), 1000).reshape(-1, 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(z, bins=50, density=True, alpha=0.4, color="gray",
                 label="Histogram Z")
        for i, gmm in enumerate(models):
            pdf = np.exp(gmm.score_samples(z_lin))
            plt.plot(
                z_lin,
                pdf,
                label=f"{components_list[i]} Gauss(s), BIC={bics[i]:.1f}",
            )
        plt.title("Gaussian Mixture Model fit to Z axis")
        plt.xlabel("Z")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)

        x = np.arange(len(components_list))
        width = 0.35
        plt.subplot(1, 2, 2)
        plt.bar(x - width / 2, bics, width, label="BIC")
        plt.bar(x + width / 2, aics, width, label="AIC")
        plt.xticks(x, [f"{n} Gauss" for n in components_list])
        plt.ylabel("Value")
        plt.title("Comparison of BIC and AIC")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        filename = "Gaussian Mixture Model fit to Z axis.png"
        filepath = os.path.join(path, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Plot saved to: {filepath}")

    return results


def find_best_gmm(chunks: Sequence[dict], name: str) -> List[dict]:
    """
    Run GMM analysis for each chunk in parallel.
    """
    os.makedirs(name, exist_ok=True)

    def worker(i: int, chunk: dict, base_name: str) -> dict:
        local_path = os.path.join(base_name, f"{i}")
        os.makedirs(local_path, exist_ok=True)
        return check_best_gmm_model(chunk["chunk"], local_path)

    results = Parallel(n_jobs=-1)(
        delayed(worker)(i, chunk, name) for i, chunk in enumerate(chunks)
    )
    return results


def analyze_gmm_fragments(
    gmm_fragments: Sequence[dict],
    path: str = ".",
) -> None:
    """
    Analyze Gaussian Mixture Model (GMM) fragments and evaluate component stability.
    """
    os.makedirs(path, exist_ok=True)

    n = len(gmm_fragments)
    if n == 0:
        print("No GMM fragments provided.")
        return

    kernel_counts = [frag["best_model"] for frag in gmm_fragments]

    anomaly_indices = [i for i, count in enumerate(kernel_counts) if count == 1]

    plt.figure(figsize=(12, 4))
    plt.plot(kernel_counts, label="Number of kernels", marker="o")
    plt.scatter(
        anomaly_indices,
        [1] * len(anomaly_indices),
        color="red",
        label="Anomalies (1 component)",
        zorder=10,
    )
    plt.title("Number of GMM components for each fragment")
    plt.xlabel("Fragment index")
    plt.ylabel("Number of components")
    plt.yticks([1, 2])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = "Number of GMM components for each fragment.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath)
    plt.close()

    gmm2 = [frag["2_components"] for frag in gmm_fragments if frag["best_model"] == 2]
    if not gmm2:
        print("No fragments with 2 components – stability analysis cannot be performed.")
        return

    means1 = [comp[0]["mean"] for comp in gmm2]
    stds1 = [comp[0]["std"] for comp in gmm2]
    means2 = [comp[1]["mean"] for comp in gmm2]
    stds2 = [comp[1]["std"] for comp in gmm2]

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].plot(means1, label="Gauss #1 (larger mean)", marker="o")
    axs[0].plot(means2, label="Gauss #2 (smaller mean)", marker="o")
    axs[0].set_title(
        "Means of GMM components in fragments (where 2 Gaussians are present)"
    )
    axs[0].set_ylabel("Mean")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(stds1, label="Gauss #1 (larger mean)", marker="o")
    axs[1].plot(stds2, label="Gauss #2 (smaller mean)", marker="o")
    axs[1].set_title("Standard deviations of GMM components")
    axs[1].set_ylabel("Std")
    axs[1].set_xlabel("Fragment index (with 2 components)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    filename = "Analysis of variance and mean in segments.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath)
    plt.close()

    mean_var1 = float(np.var(means1))
    mean_var2 = float(np.var(means2))
    std_var1 = float(np.var(stds1))
    std_var2 = float(np.var(stds2))

    print("📊 Component stability (lower variance = higher stability):")
    print(f"  Gauss #1: Var(mean) = {mean_var1:.4f}, Var(std) = {std_var1:.4f}")
    print(f"  Gauss #2: Var(mean) = {mean_var2:.4f}, Var(std) = {std_var2:.4f}")

    if mean_var1 + std_var1 < mean_var2 + std_var2:
        print("Gauss #1 (with larger mean) is more stable.")
    else:
        print("Gauss #2 (with smaller mean) is more stable.")


def get_universal_stable_profile(
    gmm_fragments: Sequence[dict],
) -> Optional[dict]:
    """
    Derive a universal stable profile from GMM fragment analyses.
    """
    upper_stable: List[dict] = []
    lower_stable: List[dict] = []

    component1_count = 0
    component2_count = 0

    for frag in gmm_fragments:
        if frag.get("best_model") != 2:
            continue

        g1, g2 = frag["2_components"]

        if g1["std"] <= g2["std"]:
            stable, unstable = g1, g2
            component1_count += 1
        else:
            stable, unstable = g2, g1
            component2_count += 1

        if stable["mean"] > unstable["mean"]:
            upper_stable.append(stable)
        else:
            lower_stable.append(stable)

    total_stable = len(upper_stable) + len(lower_stable)
    if total_stable == 0:
        print("⚠️ No fragments with 2 components – a stable trend cannot be determined.")
        return None

    if len(upper_stable) > len(lower_stable):
        dominant_group = upper_stable
        position = "above"
    elif len(upper_stable) < len(lower_stable):
        dominant_group = lower_stable
        position = "below"
    else:
        return None

    dominant_component_percent = 100.0 * len(dominant_group) / len(gmm_fragments)
    means = [g["mean"] for g in dominant_group]
    stds = [g["std"] for g in dominant_group]
    mean_of_means = float(np.mean(means))
    mean_of_stds = float(np.mean(stds))

    return {
        "mean": mean_of_means,
        "std": mean_of_stds,
        "range_min": mean_of_means - 3 * mean_of_stds,
        "range_max": mean_of_means + 3 * mean_of_stds,
        "position": position,
        "dominant_component_percent": dominant_component_percent,
        "component1_count": component1_count,
        "component2_count": component2_count,
    }


def filter_and_visualize_numpy_points(
    points_np: np.ndarray,
    stable_profile: dict,
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter points with respect to a stable vertical band and visualize the result in 3D.
    """
    os.makedirs(path, exist_ok=True)

    if points_np.ndim != 2 or points_np.shape[1] < 5:
        raise ValueError("points_np must have shape (N, 5) with [X, Y, Z, class, ID].")

    z = points_np[:, 2]
    zmin_band = stable_profile["range_min"]
    zmax_band = stable_profile["range_max"]
    position = stable_profile.get("position", "middle")

    if position == "above":
        mask_remaining = z < zmin_band
    elif position == "below":
        mask_remaining = z > zmax_band
    else:  # 'middle'
        mask_remaining = (z >= zmin_band) & (z <= zmax_band)

    mask_removed = ~mask_remaining

    removed_raw = points_np[mask_removed]
    remaining_raw = points_np[mask_remaining]

    rng = np.random.default_rng()

    def _downsample(arr: np.ndarray, max_n: int = 1000) -> np.ndarray:
        if arr.shape[0] <= max_n:
            return arr
        idx = rng.choice(arr.shape[0], max_n, replace=False)
        return arr[idx]

    removed = _downsample(removed_raw)
    remaining = _downsample(remaining_raw)

    fig = go.Figure()
    if remaining.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=remaining[:, 0],
                y=remaining[:, 1],
                z=remaining[:, 2],
                mode="markers",
                marker=dict(size=2, color="blue"),
                name="Remaining points",
            )
        )
    if removed.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=removed[:, 0],
                y=removed[:, 1],
                z=removed[:, 2],
                mode="markers",
                marker=dict(size=2, color="red", opacity=0.5),
                name="Removed points",
            )
        )

    if remaining.shape[0] + removed.shape[0] > 0:
        all_pts = np.vstack([remaining[:, :3], removed[:, :3]])
        xmin, ymin, zmin_val = np.min(all_pts, axis=0)
        xmax, ymax, zmax_val = np.max(all_pts, axis=0)

        xmid = (xmin + xmax) / 2.0
        ymid = (ymin + ymax) / 2.0
        zmid = (zmin_val + zmax_val) / 2.0

        half_range = max(xmax - xmin, ymax - ymin, zmax_val - zmin_val) / 2.0
        if half_range == 0:
            half_range = 1.0

        xrange_init = [xmid - half_range, xmid + half_range]
        yrange_init = [ymid - half_range, ymid + half_range]
        zrange_init = [zmid - half_range, zmid + half_range]
    else:
        xrange_init = [-1.0, 1.0]
        yrange_init = [-1.0, 1.0]
        zrange_init = [-1.0, 1.0]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=xrange_init),
            yaxis=dict(title="Y", range=yrange_init),
            zaxis=dict(title="Z", range=zrange_init),
            aspectmode="cube",
        ),
        title=f"Removing points relative to the stable layer ({position})",
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
    )

    inner_html = fig.to_html(include_plotlyjs='cdn', full_html=False)

    title = f"Removing points relative to the stable layer ({position})"
    note = "Każda zmiana przepisuje zakres danej osi przez Plotly.relayout()."
    custom_html = _build_html_with_axis_controls(
        title=title,
        inner_html=inner_html,
        xrange_init=xrange_init,
        yrange_init=yrange_init,
        zrange_init=zrange_init,
        note_text=note,
    )

    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    filename = f"{safe_title}.html"
    filepath = os.path.join(path, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(custom_html)

    return (
        remaining[:, :3],
        removed[:, :3],
        remaining_raw[:, 4],
        removed_raw[:, 4],
    )


def smooth_surface_from_points(
    points_np: np.ndarray,
    grid_size: int = 30,
    alpha: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a smoothed XY surface grid from scattered 3D points via local robust averaging.
    """
    if points_np.ndim != 2 or points_np.shape[1] < 3:
        raise ValueError("points_np must have at least 3 columns (X, Y, Z).")

    X = points_np[:, 0]
    Y = points_np[:, 1]
    Z = points_np[:, 2]

    x_lin = np.linspace(X.min(), X.max(), grid_size)
    y_lin = np.linspace(Y.min(), Y.max(), grid_size)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    Z_grid = np.full_like(X_grid, np.nan, dtype=float)

    dx = np.diff(x_lin).mean()
    dy = np.diff(y_lin).mean()
    rx = alpha * dx
    ry = alpha * dy

    for i in range(grid_size):
        for j in range(grid_size):
            cx, cy = X_grid[i, j], Y_grid[i, j]

            dxs = (X - cx) / rx
            dys = (Y - cy) / ry
            dist_ellipse = dxs**2 + dys**2

            inside = dist_ellipse <= 1.0
            if not np.any(inside):
                continue

            local_Z = Z[inside]
            local_XY = np.stack([X[inside], Y[inside]], axis=1)
            local_center = np.array([[cx, cy]])

            p10, p90 = np.percentile(local_Z, [10, 90])
            valid_mask = (local_Z >= p10) & (local_Z <= p90)
            if np.sum(valid_mask) < 3:
                continue

            local_Z = local_Z[valid_mask]
            local_XY = local_XY[valid_mask]

            dists = cdist(local_center, local_XY)[0]
            sigma = 0.5 * max(rx, ry)
            if sigma <= 0:
                continue

            weights = np.exp(-(dists**2) / (2 * sigma**2))
            if weights.sum() == 0:
                continue
            weights /= weights.sum()

            Z_grid[i, j] = np.sum(weights * local_Z)

    known_mask = ~np.isnan(Z_grid)
    known_points = np.stack([X_grid[known_mask], Y_grid[known_mask]], axis=1)
    known_values = Z_grid[known_mask]

    full_points = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
    Z_interp = griddata(known_points, known_values, full_points, method="cubic")
    Z_interp_grid = Z_interp.reshape(X_grid.shape)

    return X_grid, Y_grid, Z_interp_grid


def plot_surface(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z_grid: np.ndarray,
    path: str,
) -> None:
    """
    Create and save a 3D interactive surface plot from gridded data, with side controls.
    """
    os.makedirs(path, exist_ok=True)

    fig = go.Figure(
        data=[
            go.Surface(
                z=Z_grid,
                x=X_grid,
                y=Y_grid,
                colorscale="Viridis",
                opacity=0.9,
            )
        ]
    )
    fig.update_layout(
        title="Interpolated bottom surface",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
    )
    title = "Interpolated bottom surface"
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    filename = f"{safe_title}.html"
    filepath = os.path.join(path, filename)

    inner_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    xrange_init = (float(np.nanmin(X_grid)), float(np.nanmax(X_grid)))
    yrange_init = (float(np.nanmin(Y_grid)), float(np.nanmax(Y_grid)))
    zrange_init = (float(np.nanmin(Z_grid)), float(np.nanmax(Z_grid)))

    custom_html = _build_html_with_axis_controls(
        title=title,
        inner_html=inner_html,
        xrange_init=xrange_init,
        yrange_init=yrange_init,
        zrange_init=zrange_init,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(custom_html)


def prepare_and_vizualize_decission_borders(
    water_points: Sequence[np.ndarray],
    seabed_points: Sequence[np.ndarray],
    path: str,
) -> None:
    """
    Train an SVM classifier to distinguish water vs. seabed points
    and visualize the decision boundary in 3D.
    """
    os.makedirs(path, exist_ok=True)

    water_features = np.concatenate(water_points, axis=0)
    seabed_features = np.concatenate(seabed_points, axis=0)
    X = np.concatenate([water_features, seabed_features], axis=0)

    water_labels = np.ones((water_features.shape[0],), dtype=int)
    seabed_labels = np.zeros((seabed_features.shape[0],), dtype=int)
    y = np.concatenate([water_labels, seabed_labels], axis=0)

    print(f"X - {X.shape}")
    print(f"y - {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    water_extended = extend_xy_with_nearest_z(
        water_features, factor=0.6, density_ratio=1.0
    )
    seabed_extended = extend_xy_with_nearest_z(
        seabed_features, factor=0.6, density_ratio=1.0
    )
    X_train_extended = np.concatenate(
        [X_train, water_extended, seabed_extended], axis=0
    )

    water_labels_extended = np.ones(water_extended.shape[0], dtype=int)
    seabed_labels_extended = np.zeros(seabed_extended.shape[0], dtype=int)
    y_train_extended = np.concatenate(
        [y_train, water_labels_extended, seabed_labels_extended], axis=0
    )

    max_samples = min(3000, X_train_extended.shape[0])
    X_train_sampled, y_train_sampled = sample_xy(
        X_train_extended, y_train_extended, n=max_samples
    )

    clf = SVC(kernel="linear")
    clf.fit(X_train_sampled, y_train_sampled)

    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred))

    margin = 0.05
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    z_min, z_max = X[:, 2].min() - margin, X[:, 2].max() + margin

    res = 25

    x_range = np.linspace(x_min, x_max, res)
    y_range = np.linspace(y_min, y_max, res)
    z_range = np.linspace(z_min, z_max, res)

    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    decision = clf.decision_function(grid)
    decision_volume = decision.reshape((res, res, res))

    rng = np.random.default_rng()

    if water_features.shape[0] > 1000:
        idx0 = rng.choice(water_features.shape[0], 1000, replace=False)
        X0 = water_features[idx0]
    else:
        X0 = water_features

    if seabed_features.shape[0] > 1000:
        idx1 = rng.choice(seabed_features.shape[0], 1000, replace=False)
        X1 = seabed_features[idx1]
    else:
        X1 = seabed_features

    scatter0 = go.Scatter3d(
        x=X0[:, 0],
        y=X0[:, 1],
        z=X0[:, 2],
        mode="markers",
        marker=dict(size=4, color="blue"),
        name="water",
    )

    scatter1 = go.Scatter3d(
        x=X1[:, 0],
        y=X1[:, 1],
        z=X1[:, 2],
        mode="markers",
        marker=dict(size=4, color="yellow"),
        name="seabed",
    )

    isosurface = go.Isosurface(
        x=xx.flatten(),
        y=yy.flatten(),
        z=zz.flatten(),
        value=decision_volume.flatten(),
        isomin=0,
        isomax=0,
        surface_count=1,
        colorscale="Greens",
        showscale=False,
        opacity=0.5,
        name="Decision Boundary",
    )

    fig = go.Figure(data=[scatter0, scatter1, isosurface])
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="SVC Decision Boundary in 3D (Interactive Isosurface)",
        legend=dict(x=0, y=1),
        width=900,
        height=800,
    )

    x_all = np.concatenate([scatter0.x, scatter1.x, xx.flatten()])
    y_all = np.concatenate([scatter0.y, scatter1.y, yy.flatten()])
    z_all = np.concatenate([scatter0.z, scatter1.z, zz.flatten()])

    xmin_all, ymin_all, zmin_all = np.min(x_all), np.min(y_all), np.min(z_all)
    xmax_all, ymax_all, zmax_all = np.max(x_all), np.max(y_all), np.max(z_all)

    xmid = (xmin_all + xmax_all) / 2.0
    ymid = (ymin_all + ymax_all) / 2.0
    zmid = (zmin_all + zmax_all) / 2.0
    half = max(xmax_all - xmin_all, ymax_all - ymin_all, zmax_all - zmin_all) / 2.0
    if half == 0:
        half = 1.0

    xrange_init = [xmid - half, xmid + half]
    yrange_init = [ymid - half, ymid + half]
    zrange_init = [zmid - half, zmid + half]

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=xrange_init),
            yaxis=dict(range=yrange_init),
            zaxis=dict(range=zrange_init),
            aspectmode="cube",
        )
    )

    inner_html = fig.to_html(include_plotlyjs='cdn', full_html=False)

    title = "SVM_decision_plane_interpolation"
    custom_html = _build_html_with_axis_controls(
        title=title,
        inner_html=inner_html,
        xrange_init=xrange_init,
        yrange_init=yrange_init,
        zrange_init=zrange_init,
    )

    filename = "SVM_decision_plane_interpolation.html"
    filepath = os.path.join(path, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(custom_html)

    print("Saved to:", filepath)


def save_to_las(updated_array: np.ndarray, path: str) -> None:
    """
    Save a NumPy point cloud array to a LAS file.
    """
    os.makedirs(path, exist_ok=True)

    if updated_array.ndim != 2 or updated_array.shape[1] < 5:
        raise ValueError(
            "updated_array must have shape (N, 5): [x, y, z, class, ID]."
        )

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

    filename = os.path.join(path, "updated_pointcloud.las")
    las.write(filename)
    print(f"Saved as: {filename}")


def assign_classes(
    pointcloud_raw: np.ndarray,
    seabed: Sequence[int],
    water: Sequence[int],
) -> np.ndarray:
    """
    Assign classification labels to a point cloud based on point IDs.

    Default = 30, seabed = 1, water = 0.
    """
    if pointcloud_raw.ndim != 2 or pointcloud_raw.shape[1] < 5:
        raise ValueError(
            "pointcloud_raw must have shape (N, 5): [X, Y, Z, class, ID]."
        )

    updated_pointcloud = pointcloud_raw.copy()
    ids = updated_pointcloud[:, 4].astype(int)

    updated_classes = np.full(
        shape=ids.shape,
        fill_value=30,
        dtype=updated_pointcloud.dtype,
    )
    updated_classes[np.isin(ids, seabed)] = 1
    updated_classes[np.isin(ids, water)] = 0

    updated_pointcloud[:, 3] = updated_classes
    return updated_pointcloud
