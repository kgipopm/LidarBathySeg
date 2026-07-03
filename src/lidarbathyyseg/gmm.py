"""Gaussian Mixture Model fitting, analysis and stable-profile extraction."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture


def check_best_gmm_model(
    points_xyz: np.ndarray,
    output_dir: Optional[str] = None,
    saveplot: bool = True,
    gmm_max_points: int = 50_000,
    context_cloud: Optional[np.ndarray] = None,
    chunk_x_bounds: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Fit 1- and 2-component GMMs to the Z column and select the winner via BIC.

    Parameters
    ----------
    points_xyz:
        (N, ≥3) array. Only the Z column (index 2) is used.
    output_dir:
        Directory for diagnostic plots. Pass ``None`` to skip all file I/O.
    saveplot:
        Set to ``False`` to disable the BIC/AIC comparison plot entirely
        (useful in batch / testing contexts).
    gmm_max_points:
        Maximum number of Z values passed to the GMM fitter.  When the chunk
        has more points than this, a random subsample is drawn (seed=0 for
        reproducibility).  GMM fitting is a statistical procedure — 50 k
        points are more than sufficient for stable parameter estimates, so
        subsampling dramatically speeds up processing of large clouds without
        any loss in accuracy.  Set to ``0`` or ``None`` to disable.
    context_cloud:
        Optional (M, ≥3) array of the full point cloud (or a subsample of
        it) shown as background in the XZ profile panel.  When provided a
        third panel is added to the diagnostic figure.
    chunk_x_bounds:
        ``(x_lower, x_upper)`` of this chunk — used to shade the fitted
        region in the XZ profile panel.  Ignored when *context_cloud* is
        ``None``.

    Returns
    -------
    dict with keys:
        ``1_component``, ``2_components``, ``bic``, ``aic``, ``best_model``.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError(f"points_xyz must be (N, ≥3); got {points_xyz.shape}.")

    z = points_xyz[:, 2].reshape(-1, 1)

    # ── subsample Z for GMM fitting (speed) ───────────────────────────────
    if gmm_max_points and len(z) > gmm_max_points:
        rng = np.random.default_rng(0)
        z = z[rng.choice(len(z), gmm_max_points, replace=False)]

    bics, aics, models = [], [], []
    for n in (1, 2):
        gmm = GaussianMixture(n_components=n, random_state=0).fit(z)
        bics.append(float(gmm.bic(z)))
        aics.append(float(gmm.aic(z)))
        models.append(gmm)

    best_idx = int(np.argmin(bics))

    gmm1 = models[0]
    result: Dict[str, Any] = {
        "1_component": {
            "mean": float(gmm1.means_.ravel()[0]),
            "std": float(np.sqrt(gmm1.covariances_.ravel()[0])),
            "amplitude": float(gmm1.weights_[0]),
        }
    }

    gmm2 = models[1]
    means2 = gmm2.means_.flatten()
    stds2 = np.sqrt(gmm2.covariances_.reshape(2, -1).mean(axis=1))  # (2,) – independent of cov_type
    amps2 = gmm2.weights_.flatten()
    order = np.argsort(-means2)  # descending by mean

    result["2_components"] = [
        {"mean": float(means2[order[k]]), "std": float(stds2[order[k]]), "amplitude": float(amps2[order[k]])}
        for k in range(2)
    ]
    result["bic"] = {"1": bics[0], "2": bics[1]}
    result["aic"] = {"1": aics[0], "2": aics[1]}
    result["best_model"] = [1, 2][best_idx]

    if saveplot:
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        z_lin = np.linspace(z.min(), z.max(), 1000).reshape(-1, 1)

        show_profile = context_cloud is not None and context_cloud.shape[1] >= 3
        n_panels = 3 if show_profile else 2
        fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

        if show_profile:
            ax_xz, ax_hist, ax_bic = axs
            # XZ profile: full cloud (background) + this chunk (highlighted)
            ax_xz.scatter(
                context_cloud[:, 0], context_cloud[:, 2],
                c="#aaaaaa", s=0.8, alpha=0.20, rasterized=True, label="Full cloud",
            )
            ax_xz.scatter(
                points_xyz[:, 0], points_xyz[:, 2],
                c="#e63946", s=1.5, alpha=0.45, rasterized=True, label="This chunk",
            )
            if chunk_x_bounds is not None:
                ax_xz.axvspan(
                    chunk_x_bounds[0], chunk_x_bounds[1],
                    alpha=0.08, color="#e63946",
                )
            ax_xz.set_title("Pointcloud profile (XZ)")
            ax_xz.set_xlabel("X [m]")
            ax_xz.set_ylabel("Z [m]")
            ax_xz.legend(fontsize=8, markerscale=3)
            ax_xz.grid(True, alpha=0.35)
        else:
            ax_hist, ax_bic = axs

        ax_hist.hist(z, bins=50, density=True, alpha=0.4, color="gray", label="Z histogram")
        for i, (gmm_i, n) in enumerate(zip(models, (1, 2))):
            ax_hist.plot(z_lin, np.exp(gmm_i.score_samples(z_lin)),
                         label=f"{n} Gauss(s), BIC={bics[i]:.1f}")
        ax_hist.set_title("GMM fit to Z axis")
        ax_hist.set_xlabel("Z [m]")
        ax_hist.set_ylabel("Density")
        ax_hist.legend()
        ax_hist.grid(True)

        x_pos = np.arange(2)
        w = 0.35
        ax_bic.bar(x_pos - w / 2, bics, w, label="BIC")
        ax_bic.bar(x_pos + w / 2, aics, w, label="AIC")
        ax_bic.set_xticks(x_pos)
        ax_bic.set_xticklabels(["1 Gauss", "2 Gauss"])
        ax_bic.set_title("Comparison of BIC and AIC")
        ax_bic.set_xlabel("Model")
        ax_bic.set_ylabel("Score")
        ax_bic.legend()
        ax_bic.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gmm_fit_z.png"), dpi=110)
        plt.close()

    return result


_MAX_CONTEXT_DISPLAY = 5_000


def find_best_gmm(
    chunks: Sequence[Dict],
    output_dir: Optional[str] = None,
    saveplot: bool = True,
    gmm_max_points: int = 50_000,
) -> List[Dict]:
    """Run ``check_best_gmm_model`` on every chunk in parallel (joblib).

    Parameters
    ----------
    chunks:
        List of dicts from :func:`chunk_pointcloud`.
    output_dir:
        Root directory for per-chunk plots. ``None`` disables all file I/O.
    saveplot:
        Forwarded to :func:`check_best_gmm_model`.
    gmm_max_points:
        Forwarded to :func:`check_best_gmm_model` — subsample each chunk's Z
        column to at most this many points before GMM fitting.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Build a small display subsample of the full cloud for the XZ profile panel.
    # Subsampled to _MAX_CONTEXT_DISPLAY points so each joblib worker receives a
    # tiny copy rather than the full (potentially multi-GB) cloud.
    if saveplot and chunks:
        full_cloud = np.vstack([c["chunk"] for c in chunks])
        if len(full_cloud) > _MAX_CONTEXT_DISPLAY:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(full_cloud), _MAX_CONTEXT_DISPLAY, replace=False)
            display_cloud: Optional[np.ndarray] = full_cloud[idx]
        else:
            display_cloud = full_cloud
    else:
        display_cloud = None

    def _worker(i: int, chunk: Dict) -> Dict:
        local_dir = os.path.join(output_dir, str(i)) if output_dir else None
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        return check_best_gmm_model(
            chunk["chunk"], output_dir=local_dir,
            saveplot=saveplot, gmm_max_points=gmm_max_points,
            context_cloud=display_cloud,
            chunk_x_bounds=(chunk.get("lower"), chunk.get("upper")),
        )

    return Parallel(n_jobs=-1)(
        delayed(_worker)(i, c) for i, c in enumerate(chunks))


def analyze_gmm_fragments(
    gmm_fragments: Sequence[Dict],
    output_dir: Optional[str] = None,
) -> None:
    """Print stability statistics and save diagnostic plots for all GMM fragments.

    Highlights chunks where only 1 Gaussian was selected (potential anomalies)
    and reports cross-chunk variance of component means and standard deviations.
    """
    if not gmm_fragments:
        print("No GMM fragments provided.")
        return

    save = output_dir is not None
    if save:
        os.makedirs(output_dir, exist_ok=True)

    kernel_counts = [f["best_model"] for f in gmm_fragments]
    anomaly_idx = [i for i, k in enumerate(kernel_counts) if k == 1]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(kernel_counts, marker="o", label="# components")
    if anomaly_idx:
        ax.scatter(anomaly_idx, [1] * len(anomaly_idx), color="red",
                   label="Anomaly (1 component)", zorder=10)
    ax.set_title("GMM components per chunk")
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("# components")
    ax.set_yticks([1, 2])
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, "gmm_components_per_chunk.png"))
    plt.close()

    two_comp = [f["2_components"] for f in gmm_fragments if f["best_model"] == 2]
    if not two_comp:
        print("No chunks with 2 components – skipping stability analysis.")
        return

    means1 = [c[0]["mean"] for c in two_comp]
    stds1 = [c[0]["std"] for c in two_comp]
    means2 = [c[1]["mean"] for c in two_comp]
    stds2 = [c[1]["std"] for c in two_comp]

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(means1, marker="o", label="Gauss #1 (higher mean)")
    axs[0].plot(means2, marker="o", label="Gauss #2 (lower mean)")
    axs[0].set_title("Component means across chunks")
    axs[0].set_ylabel("Mean [m]")
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(stds1, marker="o", label="Gauss #1")
    axs[1].plot(stds2, marker="o", label="Gauss #2")
    axs[1].set_title("Component std across chunks")
    axs[1].set_ylabel("Std [m]")
    axs[1].set_xlabel("Chunk index (2-component only)")
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, "gmm_stability.png"))
    plt.close()

    v1 = float(np.var(means1))
    v2 = float(np.var(means2))
    vs1 = float(np.var(stds1))
    vs2 = float(np.var(stds2))

    print("GMM component stability (lower variance = more stable):")
    print(f"  Gauss #1 (higher mean): Var(mean)={v1:.4f}, Var(std)={vs1:.4f}")
    print(f"  Gauss #2 (lower mean):  Var(mean)={v2:.4f}, Var(std)={vs2:.4f}")
    stable = "#1" if (v1 + vs1) < (v2 + vs2) else "#2"
    print(f"  => Gauss {stable} is more stable (likely the seabed layer).")


def get_universal_stable_profile(
    gmm_fragments: Sequence[Dict],
    mean_var_ratio_threshold: float = 50.0,
) -> Optional[Dict]:
    """Derive the water-surface Z-band from all chunk-level GMM results.

    Physical heuristic
    ------------------
    In standard airborne LiDAR bathymetry (ALB) the water surface is
    *above* the seabed in Z.  ``check_best_gmm_model`` sorts the two GMM
    components by *descending* mean, so ``2_components[0]`` (g1) has the
    higher-Z mean and ``2_components[1]`` (g2) has the lower-Z mean.

    Default: g1 = water surface, position = 'above'.

    Z-orientation flip detection
    ----------------------------
    Some datasets have an unusual Z orientation: the water surface sits at
    *low* Z values.  In that case g2 (lower Z) is
    the water-surface band and g1 is land/terrain.

    Detection criterion — cross-chunk variance of component means::

        mean_var_ratio = Var(g1_means) / Var(g2_means)

    If this ratio exceeds ``mean_var_ratio_threshold`` (default 50), the
    higher-Z component is NOT a stable physical surface (its mean shifts
    erratically from chunk to chunk), so we flip: g2 → water surface,
    position → 'below'.

    Returns ``None`` if no 2-component chunks are available.
    """
    two_comp_frags = [f for f in gmm_fragments if f.get("best_model") == 2]
    if not two_comp_frags:
        print("Warning: no 2-component chunks – cannot determine water surface.")
        return None

    # ── Pass 1: cross-chunk variance of component means ──────────────────
    g1_means_all = [f["2_components"][0]["mean"] for f in two_comp_frags]
    g2_means_all = [f["2_components"][1]["mean"] for f in two_comp_frags]

    var_g1_means = float(np.var(g1_means_all))
    var_g2_means = float(np.var(g2_means_all))

    # Ratio: how much more variable is g1 (higher-Z) than g2 (lower-Z)?
    denom_var = max(var_g2_means, 1e-9)
    mean_var_ratio = var_g1_means / denom_var

    _ABS_INSTABILITY_FLOOR = 0.05

    flipped = (
        mean_var_ratio > mean_var_ratio_threshold
        and var_g1_means > _ABS_INSTABILITY_FLOOR
        and var_g1_means > var_g2_means
    )

    if flipped:
        print(
            f"Z-orientation flip detected: Var(g1_means)={var_g1_means:.4f}, "
            f"Var(g2_means)={var_g2_means:.4f}, ratio={mean_var_ratio:.1f} "
            f"> threshold={mean_var_ratio_threshold}. "
            f"Treating lower-Z component as water surface (position='below')."
        )
    else:
        print(
            f"Standard Z-orientation: Var(g1_means)={var_g1_means:.4f}, "
            f"Var(g2_means)={var_g2_means:.4f}, ratio={mean_var_ratio:.1f} "
            f"(threshold={mean_var_ratio_threshold}). "
            f"Higher-Z component is water surface (position='above')."
        )

    # ── Pass 2: collect water-surface component from every 2-component chunk
    water_surface_components: List[Dict] = []
    n1_stable = n2_stable = 0

    for frag in two_comp_frags:
        g1, g2 = frag["2_components"]
        water_surface_components.append(g2 if flipped else g1)
        if g1["std"] <= g2["std"]:
            n1_stable += 1
        else:
            n2_stable += 1

    total = len(water_surface_components)
    means = [g["mean"] for g in water_surface_components]
    stds  = [g["std"]  for g in water_surface_components]
    mu    = float(np.mean(means))
    sigma = float(np.mean(stds))

    position = "below" if flipped else "above"

    return {
        "mean":      mu,
        "std":       sigma,
        # Water-surface band: mu +/- 3*sigma.
        # position='above' -> seabed is BELOW range_min (filter_points uses Z < range_min).
        # position='below' -> seabed is ABOVE range_max (filter_points uses Z > range_max).
        "range_min": mu - 3 * sigma,
        "range_max": mu + 3 * sigma,
        "position":  position,
        "n_chunks_used": total,
        "dominant_component_percent": 100.0 * total / len(gmm_fragments),
        "water_more_stable_count":  n1_stable,
        "seabed_more_stable_count": n2_stable,
        "z_orientation_flipped":    flipped,
        "mean_var_ratio":           mean_var_ratio,
    }
