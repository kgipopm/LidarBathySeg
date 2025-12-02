# LidarBathySeg

**An open-source Python package for automatic water-surface and seabed
segmentation from Airborne Laser Bathymetry**

LidarBathySeg is a Python library designed for automated segmentation of
LiDAR point clouds collected in shallow-water and coastal environments.
It implements a complete workflow including filtering, coordinate
normalization, GMM-based stability detection, seabed profile extraction,
SVM modelling, and LAS export.

The library is intended for:

-   coastal geomorphology
-   shallow-water bathymetry
-   environmental monitoring
-   geospatial LiDAR research
-   reproducible scientific workflows

Processing hundreds of megabytes of LiDAR data typically takes **under
one minute** on a standard laptop thanks to efficient NumPy-based
computation.

## 🚀 Key Features

-   Automatic **water--seabed segmentation** using heuristics and
    statistical generative models.
-   **GMM stability analysis** to extract globally consistent seabed
    profiles.
-   **SVM-based classification** using automatically labelled points.
-   **3D visualization** using matplotlib and plotly.
-   Support for **LAS and LAZ** formats via `laspy[lazrs]`.
-   Modular architecture and reproducible workflows.
-   Clean `src/`-based Python package layout with `pyproject.toml`.

## 📦 Project Structure

    LidarBathySeg/
    ├── data/
    │   └── sample/
    │       └── test.las
    ├── notebooks/
    │   └── example_3d.ipynb
    ├── src/
    │   └── lidarbathyseg/
    │       ├── __init__.py
    │       ├── lidar_analysis.py
    │       └── utils.py
    ├── pyproject.toml
    ├── LICENSE
    └── README.md

## 🛠️ Installation

### 1️⃣ Clone the repository

``` bash
git clone https://github.com/<your-org-or-user>/LidarBathySeg.git
cd LidarBathySeg
```

### 2️⃣ Create and activate a virtual environment

**Windows (PowerShell):**

``` bash
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / macOS:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3️⃣ Install the package using `pyproject.toml`

``` bash
pip install -e .
```

### 4️⃣ Environment check

``` bash
python - << 'PY'
import lidarbathyseg, sys
print("LidarBathySeg OK")
print("Python:", sys.version)
print("Module path:", lidarbathyseg.__file__)
PY
```

## 📘 Usage

### ▶️ Run the example notebook

``` bash
jupyter lab notebooks/example_3d.ipynb
```

### ▶️ Minimal code example

``` python
from lidarbathyseg.utils import load_pointcloud, filter_pointcloud
from lidarbathyseg.lidar_analysis import (
    find_best_gmm,
    analyze_gmm_fragments,
    get_universal_stable_profile
)

pc = load_pointcloud("data/sample/test.las")
filtered, water_level = filter_pointcloud(pc)
gmm = find_best_gmm(filtered)

print("Processing complete.")
```

## 📂 Modules Overview

### `utils.py`

Utility functions: 
- LAS/LAZ loading
- point filtering
- rotation to principal axis
- bias correction
- chunking
- quick visualization

### `lidar_analysis.py`

Analytical core: 
- GMM model selection per chunk
- GMM fragment stability
- global seabed profile extraction
- SVM modelling and visualization
- classification and LAS export

## 🗃️ Sample Data

    data/sample/test.las

A tiny example point cloud included for testing.
Replace with your own LiDAR dataset for real workflows.

## 👥 Authors

Dr inż. Tomasz Kogut
Karol Kabała
Katedra Geodezji i Pomiarów Offshore
Politechnika Morska w Szczecinie

**Contact:**
- t.kogut@pm.szczecin.pl
- k.kabala@pm.szczecin.pl

## 📄 License

MIT License

## 🔖 Citation

``` bibtex
@article{Kogut2025LidarBathySeg,
  title={LidarBathySeg: A Python library for unsupervised segmentation of coastal and shallow-water LiDAR point clouds},
  author={Kogut, Tomasz and Kabała, Karol},
  journal={TODO},
  year={2025},
  volume={TODO},
  pages={TODO},
  doi={TODO}
}
```
