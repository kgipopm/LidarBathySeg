from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np


def build_html_with_axis_controls(
    title: str,
    inner_html: str,
    xrange_init: Sequence[float],
    yrange_init: Sequence[float],
    zrange_init: Sequence[float],
    note_text: str = "Each change rewrites the axis range via Plotly.relayout().",
) -> str:
    """Wrap a Plotly figure in a full HTML page with interactive axis-range controls.

    Returns the complete HTML string (write it to a file yourself).
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
  #plot-container {{ flex: 1 1 auto; min-height: 0; min-width: 0; height: 100%; }}
  #plot-inner {{ height: 100%; width: 100%; }}
  #plot-inner .js-plotly-plot,
  #plot-inner .plotly-graph-div {{ height: 100% !important; width: 100% !important; }}
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
  .ctrl-block h3 {{ margin: 0 0 0.5rem 0; font-size: 14px; font-weight: 600; }}
  .row {{ display: flex; align-items: center; gap: .5rem; margin-bottom: .5rem; }}
  .row label {{ width: 45px; font-weight: 600; font-size: 13px; }}
  .row input[type=range] {{ flex: 1; }}
  .row input[type=number] {{ width: 70px; }}
  button {{
    width: 100%; padding: .5rem .75rem; font-size: 14px; font-weight: 600;
    border-radius: 6px; border: 1px solid #888; background: #eee; cursor: pointer;
  }}
  button:hover {{ background: #ddd; }}
  small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<div id="plot-wrapper">
  <div id="plot-container">
    <div id="plot-inner">{inner_html}</div>
  </div>
</div>
<aside id="controls">
  <div class="ctrl-block">
    <h3>Reset view</h3>
    <button id="resetBtn">Reset to cube</button>
  </div>
  <div class="ctrl-block">
    <h3>Axis ranges</h3>
    <div class="row"><label>X min</label><input id="xMinNum" type="number" step="0.1" value="{x0:.2f}"></div>
    <div class="row">
      <label>X max</label>
      <input id="xMaxRange" type="range" min="{x0:.2f}" max="{x1*2:.2f}" step="0.1" value="{x1:.2f}">
      <input id="xMaxNum" type="number" step="0.1" value="{x1:.2f}">
    </div>
    <div class="row"><label>Y min</label><input id="yMinNum" type="number" step="0.1" value="{y0:.2f}"></div>
    <div class="row">
      <label>Y max</label>
      <input id="yMaxRange" type="range" min="{y0:.2f}" max="{y1*2:.2f}" step="0.1" value="{y1:.2f}">
      <input id="yMaxNum" type="number" step="0.1" value="{y1:.2f}">
    </div>
    <div class="row"><label>Z min</label><input id="zMinNum" type="number" step="0.1" value="{z0:.2f}"></div>
    <div class="row">
      <label>Z max</label>
      <input id="zMaxRange" type="range" min="{z0:.2f}" max="{z1*2:.2f}" step="0.1" value="{z1:.2f}">
      <input id="zMaxNum" type="number" step="0.1" value="{z1:.2f}">
    </div>
    <small>{note_text}</small>
  </div>
</aside>
<script>
const plotDiv = document.querySelector('#plot-inner').querySelector('div.js-plotly-plot');
const initRanges = {{ x: [{x0}, {x1}], y: [{y0}, {y1}], z: [{z0}, {z1}] }};

function setAxisRange(axis, minVal, maxVal) {{
  const u = {{}};
  u[`scene.${{axis}}axis.range`] = [minVal, maxVal];
  Plotly.relayout(plotDiv, u);
}}

const xMinNum = document.getElementById('xMinNum');
const xMaxRange = document.getElementById('xMaxRange');
const xMaxNum = document.getElementById('xMaxNum');
function updateX() {{
  const a = parseFloat(xMinNum.value), b = parseFloat(xMaxNum.value);
  if (!isNaN(a) && !isNaN(b) && b > a) setAxisRange('x', a, b);
}}
xMinNum.addEventListener('change', updateX);
xMaxRange.addEventListener('input', e => {{ xMaxNum.value = e.target.value; updateX(); }});
xMaxNum.addEventListener('change', () => {{ xMaxRange.value = xMaxNum.value; updateX(); }});

const yMinNum = document.getElementById('yMinNum');
const yMaxRange = document.getElementById('yMaxRange');
const yMaxNum = document.getElementById('yMaxNum');
function updateY() {{
  const a = parseFloat(yMinNum.value), b = parseFloat(yMaxNum.value);
  if (!isNaN(a) && !isNaN(b) && b > a) setAxisRange('y', a, b);
}}
yMinNum.addEventListener('change', updateY);
yMaxRange.addEventListener('input', e => {{ yMaxNum.value = e.target.value; updateY(); }});
yMaxNum.addEventListener('change', () => {{ yMaxRange.value = yMaxNum.value; updateY(); }});

const zMinNum = document.getElementById('zMinNum');
const zMaxRange = document.getElementById('zMaxRange');
const zMaxNum = document.getElementById('zMaxNum');
function updateZ() {{
  const a = parseFloat(zMinNum.value), b = parseFloat(zMaxNum.value);
  if (!isNaN(a) && !isNaN(b) && b > a) setAxisRange('z', a, b);
}}
zMinNum.addEventListener('change', updateZ);
zMaxRange.addEventListener('input', e => {{ zMaxNum.value = e.target.value; updateZ(); }});
zMaxNum.addEventListener('change', () => {{ zMaxRange.value = zMaxNum.value; updateZ(); }});

document.getElementById('resetBtn').addEventListener('click', () => {{
  xMinNum.value = initRanges.x[0]; xMaxNum.value = initRanges.x[1]; xMaxRange.value = initRanges.x[1];
  yMinNum.value = initRanges.y[0]; yMaxNum.value = initRanges.y[1]; yMaxRange.value = initRanges.y[1];
  zMinNum.value = initRanges.z[0]; zMaxNum.value = initRanges.z[1]; zMaxRange.value = initRanges.z[1];
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


def plot_xz_classification(
    raw_cloud: np.ndarray,
    seabed_ids: np.ndarray,
    water_ids: np.ndarray,
    output_dir: str,
    filename: str = "xz_classification.png",
    max_points: int = 8_000,
    title: str = "",
    stable_profile: Optional[dict] = None,
    z_bias: float = 0.0,
) -> str:
    """Save an XZ scatter of the classified point cloud.

    Uses the *original* (pre-processing) coordinates so that Z represents
    true depth.  No ground-truth labels are required.

    Colour scheme
    -------------
    blue   – predicted water
    sienna – predicted seabed
    grey   – unclassified (not assigned to either class)

    Green dashed band: water-surface ±3σ zone, derived from the 2nd–98th
    percentile of predicted-water Z values in the original coordinate space.

    Parameters
    ----------
    raw_cloud:
        (N, ≥5) array [X, Y, Z, class, ID] in original coordinates.
    seabed_ids, water_ids:
        1-D integer arrays of point IDs from the segmentation pipeline.
    output_dir:
        Directory where the PNG is saved.
    filename:
        Output file name (default: ``xz_classification.png``).
    max_points:
        Maximum number of points rendered (random subsample for speed).
    title:
        Optional title suffix appended to the figure title.

    Returns
    -------
    str
        Absolute path of the saved PNG file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_xz_classification.")

    os.makedirs(output_dir, exist_ok=True)

    # ── subsample for rendering speed ─────────────────────────────────
    rng = np.random.default_rng(42)
    n   = min(max_points, len(raw_cloud))
    idx = rng.choice(len(raw_cloud), n, replace=False)
    sub = raw_cloud[idx]             # (n, ≥5)

    sub_ids = sub[:, 4].astype(int)
    sb_set  = set(seabed_ids.astype(int))
    w_set   = set(water_ids.astype(int))

    pred_sb   = np.isin(sub_ids, list(sb_set))
    pred_w    = np.isin(sub_ids, list(w_set))
    unclassed = ~pred_sb & ~pred_w

    # ── water-surface band: stable_profile back-projected to original Z ──
    vis_profile: Optional[dict] = None
    if stable_profile is not None:
        # stable_profile lives in preprocessed space (Z_proc = Z_orig - z_bias)
        mu  = stable_profile["mean"]      + z_bias
        rmi = stable_profile["range_min"] + z_bias
        rma = stable_profile["range_max"] + z_bias
        vis_profile = {"mean": mu, "range_min": rmi, "range_max": rma}
    else:
        all_ids    = raw_cloud[:, 4].astype(int)
        water_mask = np.isin(all_ids, list(w_set))
        if water_mask.sum() > 0:
            wz  = raw_cloud[water_mask, 2]
            mu  = float(np.mean(wz))
            sig = float(np.std(wz))
            vis_profile = {"mean": mu, "range_min": mu - 3 * sig, "range_max": mu + 3 * sig}

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))

    if unclassed.any():
        ax.scatter(sub[unclassed, 0], sub[unclassed, 2],
                   c="#bbbbbb", alpha=0.20, s=2,
                   rasterized=True, label="Unclassified", zorder=1)
    ax.scatter(sub[pred_w,  0], sub[pred_w,  2],
               c="#3a86ff", alpha=0.22, s=3,
               rasterized=True, label=f"Water ({len(w_set):,} pts)", zorder=2)
    ax.scatter(sub[pred_sb, 0], sub[pred_sb, 2],
               c="#b5451b", alpha=0.60, s=4,
               rasterized=True, label=f"Seabed ({len(sb_set):,} pts)", zorder=3)

    if vis_profile:
        ax.axhspan(vis_profile["range_min"], vis_profile["range_max"],
                   alpha=0.10, color="#2dc653",
                   label=(f"Water band μ±3σ "
                          f"[{vis_profile['range_min']:.2f}, "
                          f"{vis_profile['range_max']:.2f}]"))
        ax.axhline(vis_profile["mean"], color="#2dc653",
                   linewidth=1.2, linestyle="--", alpha=0.8)

    heading = f"XZ classification – {title}" if title else "XZ classification"
    ax.set_title(heading, fontsize=10)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.legend(loc="lower right", fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return os.path.abspath(out_path)
