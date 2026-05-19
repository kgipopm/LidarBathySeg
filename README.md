# LidarBathySeg

**An open-source Python package for automatic water-surface and seabed
segmentation from Airborne Laser Bathymetry**

LidarBathySeg is a Python library for automated segmentation of LiDAR point
clouds collected in shallow-water and coastal environments. It implements a
complete workflow: filtering, coordinate normalisation, adaptive spatial
chunking, GMM-based seabed-profile detection, seabed surface reconstruction,
and LAS export.

The library is intended for:

- coastal geomorphology
- shallow-water bathymetry
- environmental monitoring
- geospatial LiDAR research
- reproducible scientific workflows

Processing time scales with file size. Small files (~200 k points) finish in
under 30 seconds; large files (tens of millions of points) typically take
several minutes. GMM fitting is subsampled to at most 50 000 points per chunk
so the statistical step stays fast even on 60 M-point clouds.

---

## Key Features

- Automatic **water–seabed segmentation** using Gaussian Mixture Models fitted
  independently to spatial chunks.
- **Adaptive chunk-count selection** — geometric-scale search optimising GMM
  stability across spatial slices (no manual tuning required).
- **Z-axis orientation detection** — automatically distinguishes standard ALB
  orientation (water above seabed) from inverted datasets (e.g. coastal scenes
  where cliffs dominate the upper Z range).
- **GMM stability analysis** to extract a globally consistent water-surface
  Z-band and derive the seabed classification boundary.
- **Unclassified-point reclassification** — points removed by pre-processing
  or in under-size chunks are re-assigned using the water-surface boundary
  derived from the already-classified points.
- **Seabed surface reconstruction** via Delaunay-interpolated smooth mesh.
- **XZ classification plot** — quick visual QC showing water / seabed /
  water-surface band in the cross-shore plane.
- **Interactive 3-D visualisation** using Plotly (saved as self-contained HTML).
- Support for **LAS and LAZ** formats via `laspy[lazrs]`.
- Fluent high-level `SegmentationPipeline` API or low-level building blocks.

---

## Project Structure

```
LidarBathySeg/
├── src/
│   └── lidarbathyseg/
│       ├── __init__.py         # public API + backward-compat aliases
│       ├── preprocessing.py    # load, filter, rotate, remove_bias
│       ├── chunking.py         # chunk_pointcloud, select_optimal_chunk_count
│       ├── gmm.py              # check_best_gmm_model, find_best_gmm,
│       │                       # analyze_gmm_fragments, get_universal_stable_profile
│       ├── segmentation.py     # filter_points, assign_classes
│       ├── surface.py          # smooth_surface_from_points, plot_surface
│       ├── visualization.py    # plot_xz_classification, build_html_with_axis_controls
│       ├── classification.py   # train_svm_classifier  (optional, standalone)
│       ├── io.py               # save_to_las, save_results_to_original_las
│       └── pipeline.py         # SegmentationPipeline (fluent API)
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org-or-user>/LidarBathySeg.git
cd LidarBathySeg
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install the package

```bash
pip install -e .
```

All dependencies are declared in `pyproject.toml` and installed automatically.

### 4. LAZ support

LAZ decompression requires the `lazrs` backend.  If it was not pulled in
automatically, install it explicitly:

```bash
pip install "laspy[lazrs]"
```

### 5. Verify

```bash
python - << 'PY'
import lidarbathyseg, sys
print("LidarBathySeg OK")
print("Python:", sys.version)
print("Module:", lidarbathyseg.__file__)
PY
```

---

## Quick Start

### One-liner pipeline

```python
from lidarbathyseg import SegmentationPipeline

pipe = SegmentationPipeline(output_dir="results", auto_chunks=True)
pipe.run("data/sample/test.las")   # load → preprocess → chunk → GMM → segment → XZ plot
pipe.save_las()
pipe.build_surface()
```

### Step-by-step API

```python
import lidarbathyseg as lbs
import numpy as np

# 1 – load
raw = lbs.load_pointcloud("data/sample/test.las")

# 2 – preprocess
filtered        = lbs.filter_pointcloud(raw)
rotated, slope  = lbs.rotate_to_principal_axis(filtered)
cloud           = lbs.remove_bias(rotated)

# 3 – chunk (auto-selects optimal bin count)
n_chunks, scores = lbs.select_optimal_chunk_count(cloud, verbose=True)
chunks           = lbs.chunk_pointcloud(cloud, x_bins=n_chunks)

# 4 – GMM
gmm_results    = lbs.find_best_gmm(chunks, output_dir="results/gmm")
stable_profile = lbs.get_universal_stable_profile(gmm_results)

# 5 – segment
seabed_ids, water_ids = [], []
for chunk in chunks:
    _, _, ids_s, ids_w = lbs.filter_points(chunk["chunk"], stable_profile)
    seabed_ids.append(ids_s)
    water_ids.append(ids_w)

classified = lbs.assign_classes(raw, np.concatenate(seabed_ids),
                                     np.concatenate(water_ids))
lbs.save_to_las(classified, "results")
```

---

## Module Reference

### `preprocessing.py`

| Function | Description |
|---|---|
| `load_pointcloud(path)` | Read LAS/LAZ → `(N, 5)` array `[x, y, z, class, id]` |
| `filter_pointcloud(cloud, z_low_pct, z_high_pct)` | Remove Z outliers (percentile clip) |
| `rotate_to_principal_axis(cloud)` | Align dominant elongation with X axis via PCA |
| `remove_bias(cloud)` | Shift min(x, y, z) to 0 |

### `chunking.py`

| Function | Description |
|---|---|
| `chunk_pointcloud(cloud, x_bins)` | Split cloud into `x_bins` equal-width X slices |
| `select_optimal_chunk_count(cloud, ...)` | Adaptive bin-count search (see below) |

#### Adaptive chunk-count selection

`select_optimal_chunk_count` evaluates `n_probe` (default 8) candidates on a
**geometric scale** between `min_bins` and `max_bins`. For each candidate it
assigns a score:

```
score = frac_2comp × stability

frac_2comp  = fraction of chunks where BIC favours a 2-component GMM
stability   = 1 / (1 + var(dominant-component means across chunks))
```

`frac_2comp` rewards configurations where each slice clearly shows two layers
(water and seabed). `stability` penalises configurations where the chunk means
jump erratically (too many bins → too little data per bin). The product peaks
at a natural optimum, requiring no manual input.

### `gmm.py`

| Function | Description |
|---|---|
| `check_best_gmm_model(chunk, gmm_max_points)` | Fit 1- and 2-component GMMs to the Z column; pick winner via BIC. Subsamples to `gmm_max_points` (default 50 000) for speed. |
| `find_best_gmm(chunks, output_dir, gmm_max_points)` | Run `check_best_gmm_model` over all chunks in parallel (joblib) |
| `analyze_gmm_fragments(results)` | Print cross-chunk variance report; save stability plots |
| `get_universal_stable_profile(results)` | Extract consensus water-surface Z-band with Z-axis orientation detection (see below) |

#### Z-axis orientation detection

`get_universal_stable_profile` uses a two-step heuristic to detect whether the
water surface sits at high Z (standard ALB) or low Z (inverted datasets such
as coastal scenes where cliffs dominate the upper Z range):

1. **Cross-chunk mean variance ratio** —
   `Var(g1_means) / Var(g2_means) > 50`:
   the higher-Z component's means vary wildly across chunks (terrain heights
   differ along X), while the lower-Z component (sea level) stays constant.

2. **Absolute instability guard** —
   `Var(g1_means) > 0.05 m²`:
   prevents false flips when both components are near-perfectly stable
   (e.g. flat seabed where both variances are near zero but their ratio is
   numerically large).

When both conditions are met the function sets `position = 'below'`
(water is at low Z, seabed above the water band). Otherwise
`position = 'above'` (standard: seabed below the water band).

### `segmentation.py`

| Function | Description |
|---|---|
| `filter_points(chunk, profile)` | Split chunk into seabed / water point sets using the stable Z-band; respects `profile["position"]` |
| `assign_classes(cloud, seabed_ids, water_ids)` | Stamp class labels (**1 = seabed, 0 = water, 30 = unclassified**) onto the raw cloud |

### `surface.py`

| Function | Description |
|---|---|
| `smooth_surface_from_points(pts)` | Delaunay triangulation + bilinear interpolation → regular `(Xg, Yg, Zg)` grid |
| `plot_surface(Xg, Yg, Zg, output_dir)` | Save interactive Plotly 3-D surface as HTML |

### `visualization.py`

| Function | Description |
|---|---|
| `plot_xz_classification(raw_cloud, seabed_ids, water_ids, ...)` | Save XZ scatter PNG: blue = water, sienna = seabed, green band = water-surface ±3σ zone |
| `build_html_with_axis_controls(fig)` | Wrap a Plotly figure with JS axis-toggle controls |

### `io.py`

| Function | Description |
|---|---|
| `save_to_las(cloud, output_dir)` | Write classified cloud as `updated_pointcloud.las` |
| `save_results_to_original_las(cloud, original_path, output_dir)` | Merge classification labels into the original LAS file |

### `classification.py` *(optional, standalone)*

| Function | Description |
|---|---|
| `train_svm_classifier(water_pts, seabed_pts, output_dir)` | Train a linear SVM on GMM-segmented point sets; returns metrics dict |

> **Note:** This module is not invoked by `SegmentationPipeline`. It can be
> used as a post-processing step or alternative classifier on top of the
> GMM-derived labels.

### `pipeline.py` — `SegmentationPipeline`

Fluent wrapper around the full workflow. Every method returns `self` for
chaining.

```python
SegmentationPipeline(
    output_dir      = "output",
    n_chunks        = None,       # fixed bin count; overrides auto_chunks
    auto_chunks     = True,       # run select_optimal_chunk_count
    save_chunk_plots= True,       # save per-chunk GMM plots
    gmm_max_points  = 50_000,     # max Z-points subsampled before GMM fit
)
  .load(file_path)          # load LAS/LAZ from disk
  .load_array(numpy_array)  # or supply an existing (N, 5) array
  .preprocess()             # filter → rotate → remove_bias
  .chunk()                  # spatial X-slices (auto or fixed count)
  .fit_gmm()                # per-chunk GMM + universal stable profile
  .segment()                # classify chunks; reclassify unprocessed points
  .visualize_xz()           # save XZ classification PNG
  .build_surface()          # reconstruct seabed surface (optional)
  .save_las()               # write updated_pointcloud.las

# convenience — runs load → preprocess → chunk → fit_gmm → segment → visualize_xz:
  .run(file_path)
```

**Key attributes after `.segment()`:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `seabed_ids` | `np.ndarray` | Point IDs classified as seabed |
| `water_ids` | `np.ndarray` | Point IDs classified as water |
| `stable_profile` | `dict` | Water-surface band: `mean`, `std`, `range_min`, `range_max`, `position`, `z_orientation_flipped` |
| `chunk_scores` | `dict` | Score map from adaptive chunk-count selection |
| `assigned_cloud` | `np.ndarray` | Full cloud with class labels stamped in column 3 |

---

## Authors

- **Tomasz Kogut** — Maritime University of Szczecin · t.kogut@pm.szczecin.pl
- **Karol Kabala** — Maritime University of Szczecin · k.kabala@pm.szczecin.pl
- **Malgorzata Jarzabek-Rychard** — Wroclaw University of Environmental and Life Sciences · malgorzata.jarzabek-rychard@upwr.edu.pl
- **Arkadiusz Tomczak** — Maritime University of Szczecin · a.tomczak@pm.szczecin.pl

---

## License

MIT License

---

## Citation

```bibtex
@article{Kogut2026LidarBathySeg,
  title   = {LidarBathySeg: An open-source Python software for water surface--seabed segmentation from Airborne Laser Bathymetry},
  author  = {Kogut, Tomasz and Kabala, Karol and Jarzabek-Rychard, Malgorzata and Tomczak, Arkadiusz},
  journal = {TODO},
  year    = {2026},
  volume  = {TODO},
  pages   = {TODO},
  doi     = {TODO}
}
```
