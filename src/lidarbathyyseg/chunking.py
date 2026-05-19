from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def chunk_pointcloud(
    cloud: np.ndarray,
    x_bins: int = 5,
) -> List[Dict]:
    """Split *cloud* into *x_bins* equal-width slices along the X-axis.

    Returns a list of dicts with keys: ``chunk``, ``lower``, ``upper``, ``shape``.
    """
    if cloud.shape[1] == 0:
        raise ValueError("`cloud` must have at least 1 column.")
    if x_bins <= 0:
        raise ValueError("`x_bins` must be a positive integer.")

    x_min = float(cloud[:, 0].min())
    x_max = float(cloud[:, 0].max())
    dx = (x_max - x_min) / x_bins

    chunks: List[Dict] = []
    for i in range(x_bins):
        lower = x_min + i * dx
        upper = x_min + (i + 1) * dx if i < x_bins - 1 else x_max

        if i < x_bins - 1:
            mask = (cloud[:, 0] >= lower) & (cloud[:, 0] < upper)
        else:
            mask = (cloud[:, 0] >= lower) & (cloud[:, 0] <= upper)

        c = cloud[mask]
        chunks.append({"chunk": c, "lower": lower, "upper": upper, "shape": c.shape})

    return chunks


def select_optimal_chunk_count(
    cloud: np.ndarray,
    min_bins: int = 3,
    max_bins: int = 30,
    min_points_per_chunk: int = 50,
    n_probe: int = 8,
    verbose: bool = False,
) -> Tuple[int, Dict[int, Dict]]:
    """Adaptively select the number of spatial chunks that maximises GMM stability.

    The stability score for a given *n_bins* is::

        score = frac_2comp × 1 / (1 + var_of_means)

    where *frac_2comp* is the fraction of chunks best described by 2 Gaussians
    and *var_of_means* is the cross-chunk variance of the **more stable component mean**
    (whichever of the two GMM components has lower cross-chunk mean variance — always the
    water surface, regardless of Z orientation).
    Candidates are drawn on a geometric scale so both small and large values are
    probed without exhaustive search.

    Parameters
    ----------
    cloud:
        Preprocessed point cloud (N, ≥3).
    min_bins, max_bins:
        Inclusive search bounds.
    min_points_per_chunk:
        Chunks with fewer points are skipped during scoring.
    n_probe:
        Number of candidate bin-counts to evaluate (geometric sequence).
    verbose:
        Print per-candidate scores.

    Returns
    -------
    best_n : int
        Recommended number of chunks.
    scores : dict
        Mapping ``n_bins → score_dict`` for every evaluated candidate.
    """
    from .gmm import check_best_gmm_model

    N = cloud.shape[0]
    max_bins = min(max_bins, N // max(1, min_points_per_chunk))
    max_bins = max(min_bins, max_bins)

    if max_bins <= min_bins:
        return min_bins, {}

    candidates = np.unique(
        np.round(np.geomspace(min_bins, max_bins, n_probe)).astype(int)
    ).tolist()

    scores: Dict[int, Dict] = {}
    best_n = candidates[0]
    best_score = -np.inf

    for n_bins in candidates:
        chunks = chunk_pointcloud(cloud, x_bins=n_bins)
        valid = [c for c in chunks if len(c["chunk"]) >= min_points_per_chunk]

        if len(valid) < 2:
            scores[n_bins] = {"score": 0.0, "reason": "too_few_valid_chunks"}
            continue

        gmm_results = []
        for c in valid:
            try:
                r = check_best_gmm_model(c["chunk"], output_dir=None, saveplot=False,
                                         gmm_max_points=50_000)
                gmm_results.append(r)
            except Exception:
                continue

        if not gmm_results:
            scores[n_bins] = {"score": 0.0, "reason": "gmm_failed"}
            continue

        n_two = sum(1 for r in gmm_results if r["best_model"] == 2)
        frac_two = n_two / len(gmm_results)

        if n_two >= 2:
            two_comp = [r for r in gmm_results if r["best_model"] == 2]
            means_g1 = [r["2_components"][0]["mean"] for r in two_comp]
            means_g2 = [r["2_components"][1]["mean"] for r in two_comp]
            var_g1 = float(np.var(means_g1))
            var_g2 = float(np.var(means_g2))
            # Use the more stable component (lower cross-chunk variance of means).
            # The water surface is always flatter than the seabed/terrain regardless
            # of Z orientation, so min() picks water without needing to know which
            # component sits at higher Z.
            var_means = min(var_g1, var_g2)
        else:
            var_means = np.inf

        stability = 1.0 / (1.0 + var_means)
        score = frac_two * stability

        entry = {
            "score": score,
            "frac_two": frac_two,
            "var_means": var_means if np.isfinite(var_means) else None,
            "stability": stability,
            "n_valid_chunks": len(valid),
            "n_gmm_fitted": len(gmm_results),
        }
        scores[n_bins] = entry

        if verbose:
            print(
                f"  n_bins={n_bins:3d} | frac_2comp={frac_two:.2f} "
                f"| var_means={var_means:.4f} | score={score:.4f}"
            )

        if score > best_score:
            best_score = score
            best_n = n_bins

    if verbose:
        print(f"=> Selected n_bins={best_n} (score={best_score:.4f})")

    return best_n, scores
