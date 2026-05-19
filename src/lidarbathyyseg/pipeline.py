from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .chunking import chunk_pointcloud, select_optimal_chunk_count
from .gmm import analyze_gmm_fragments, find_best_gmm, get_universal_stable_profile
from .io import save_to_las
from .preprocessing import filter_pointcloud, remove_bias, rotate_to_principal_axis
from .segmentation import assign_classes, filter_points
from .surface import plot_surface, smooth_surface_from_points
from .visualization import plot_xz_classification


class SegmentationPipeline:
    """End-to-end LiDAR bathymetry segmentation in a fluent, chainable API.

    Typical usage
    -------------
    >>> pipe = SegmentationPipeline(output_dir="results", auto_chunks=True)
    >>> pipe.load("data/sample/test.las").preprocess().chunk().fit_gmm().segment()
    >>> pipe.save_las()

    Or as a one-liner:

    >>> pipe = SegmentationPipeline("results").run("data/sample/test.las")

    Attributes
    ----------
    n_chunks : int or None
        Number of X-axis bins. Set automatically when ``auto_chunks=True``.
    chunk_scores : dict
        Score map returned by :func:`select_optimal_chunk_count`.
    stable_profile : dict or None
        Universal seabed Z-band derived from GMM fragments.
    seabed_ids, water_ids : np.ndarray
        Concatenated point IDs classified as seabed / water.
    """

    def __init__(
        self,
        output_dir: str = "output",
        n_chunks: Optional[int] = None,
        auto_chunks: bool = True,
        save_chunk_plots: bool = True,
        gmm_max_points: int = 50_000,
    ) -> None:
        self.output_dir = output_dir
        self.n_chunks = n_chunks
        self.auto_chunks = auto_chunks
        self.save_chunk_plots = save_chunk_plots
        self.gmm_max_points = gmm_max_points

        self.raw_cloud: Optional[np.ndarray] = None
        self.processed_cloud: Optional[np.ndarray] = None
        self.chunks: Optional[List[dict]] = None
        self.chunk_scores: dict = {}
        self.gmm_results: Optional[List[dict]] = None
        self.stable_profile: Optional[dict] = None
        self._seabed_id_lists: List[np.ndarray] = []
        self._water_id_lists: List[np.ndarray] = []
        self.seabed_ids: Optional[np.ndarray] = None
        self.water_ids: Optional[np.ndarray] = None
        self.assigned_cloud: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Step 1 – load
    # ------------------------------------------------------------------

    def load(self, file_path: str) -> "SegmentationPipeline":
        """Load a LAS file into ``self.raw_cloud``."""
        from .preprocessing import load_pointcloud
        self.raw_cloud = load_pointcloud(file_path)
        print(f"Loaded {self.raw_cloud.shape[0]:,} points from {file_path!r}")
        return self

    def load_array(self, cloud: np.ndarray) -> "SegmentationPipeline":
        """Use an existing (N, 5) numpy array instead of a LAS file."""
        self.raw_cloud = cloud
        return self

    # ------------------------------------------------------------------
    # Step 2 – preprocess
    # ------------------------------------------------------------------

    def preprocess(self) -> "SegmentationPipeline":
        """Filter outliers, rotate to principal axis and remove coordinate bias."""
        if self.raw_cloud is None:
            raise RuntimeError("Call .load() before .preprocess().")
        filtered = filter_pointcloud(self.raw_cloud)
        rotated, _ = rotate_to_principal_axis(filtered)
        # Store Z bias before remove_bias shifts it to 0; rotation is XY-only so Z is unchanged
        self._z_bias = float(rotated[:, 2].min())
        self.processed_cloud = remove_bias(rotated)
        print(f"After preprocessing: {self.processed_cloud.shape[0]:,} points.")
        return self

    # ------------------------------------------------------------------
    # Step 3 – chunk
    # ------------------------------------------------------------------

    def chunk(self) -> "SegmentationPipeline":
        """Partition the cloud into X-axis slices.

        If ``auto_chunks=True`` and ``n_chunks`` is not set, runs
        :func:`select_optimal_chunk_count` to pick the best bin count.
        """
        if self.processed_cloud is None:
            raise RuntimeError("Call .preprocess() before .chunk().")

        if self.auto_chunks and self.n_chunks is None:
            print("Selecting optimal chunk count …")
            self.n_chunks, self.chunk_scores = select_optimal_chunk_count(
                self.processed_cloud, verbose=True
            )

        n = self.n_chunks or 5
        self.chunks = chunk_pointcloud(self.processed_cloud, x_bins=n)
        print(f"Created {len(self.chunks)} chunks (n_chunks={n}).")
        return self

    # ------------------------------------------------------------------
    # Step 4 – GMM analysis
    # ------------------------------------------------------------------

    def fit_gmm(self) -> "SegmentationPipeline":
        """Fit GMMs to every chunk and derive the universal stable profile."""
        if self.chunks is None:
            raise RuntimeError("Call .chunk() before .fit_gmm().")

        gmm_dir = os.path.join(self.output_dir, "gmm") if self.save_chunk_plots else None
        self.gmm_results = find_best_gmm(
            self.chunks, output_dir=gmm_dir, saveplot=self.save_chunk_plots,
            gmm_max_points=self.gmm_max_points,
        )
        analyze_gmm_fragments(self.gmm_results, output_dir=self.output_dir if self.save_chunk_plots else None)
        self.stable_profile = get_universal_stable_profile(self.gmm_results)

        if self.stable_profile:
            p = self.stable_profile
            print(
                f"Stable profile: mean={p['mean']:.3f}, std={p['std']:.3f}, "
                f"position={p['position']!r}, "
                f"dominant={p['dominant_component_percent']:.1f}%"
            )
        else:
            print("Warning: could not determine a stable profile.")
        return self

    # ------------------------------------------------------------------
    # Step 5 – segment
    # ------------------------------------------------------------------

    def segment(self) -> "SegmentationPipeline":
        """Filter each chunk into seabed / water point sets, then reclassify stragglers.

        After the main GMM-based pass, some raw_cloud points remain unclassified:
        they were either removed by ``filter_pointcloud`` (pre-processing outliers)
        or fell into a chunk that was too small (< 5 pts) to be processed.  These
        are reassigned using the water-surface Z boundary derived from the already-
        classified water points in the **original** (pre-processing) coordinate
        space, honouring the ``position`` field of the stable profile so that the
        heuristic works regardless of Z-axis orientation.
        """
        if self.stable_profile is None:
            raise RuntimeError("Call .fit_gmm() before .segment().")

        self._seabed_id_lists.clear()
        self._water_id_lists.clear()

        for i, chunk in enumerate(self.chunks):
            if len(chunk["chunk"]) < 5:
                continue
            chunk_dir = (
                os.path.join(self.output_dir, "chunks", f"chunk_{i:03d}")
                if self.save_chunk_plots else None
            )
            _, _, ids_s, ids_w = filter_points(
                chunk["chunk"],
                self.stable_profile,
                output_dir=chunk_dir,
                save_html=self.save_chunk_plots,
            )
            self._seabed_id_lists.append(ids_s)
            self._water_id_lists.append(ids_w)

        self.seabed_ids = np.concatenate(self._seabed_id_lists) if self._seabed_id_lists else np.array([])
        self.water_ids = np.concatenate(self._water_id_lists) if self._water_id_lists else np.array([])
        print(f"Segmented: {len(self.seabed_ids):,} seabed, {len(self.water_ids):,} water points.")

        all_raw_ids = self.raw_cloud[:, 4].astype(int)
        classified_set = set(self.seabed_ids.astype(int)) | set(self.water_ids.astype(int))
        unclassified_mask = ~np.isin(all_raw_ids, list(classified_set))
        n_unc = int(unclassified_mask.sum())

        if n_unc > 0 and len(self.water_ids) > 0:
            water_in_raw = np.isin(all_raw_ids, list(set(self.water_ids.astype(int))))
            wz = self.raw_cloud[water_in_raw, 2]

            position = self.stable_profile.get("position", "above")
            if position == "above":
                # Water surface is at HIGH Z; seabed is BELOW → boundary = lower edge
                z_boundary = float(np.percentile(wz, 2))
                is_seabed = self.raw_cloud[unclassified_mask, 2] < z_boundary
            else:
                # Water surface is at LOW Z; seabed is ABOVE → boundary = upper edge
                z_boundary = float(np.percentile(wz, 98))
                is_seabed = self.raw_cloud[unclassified_mask, 2] > z_boundary

            unclassified_pts = self.raw_cloud[unclassified_mask]
            new_seabed_ids = unclassified_pts[is_seabed,  4].astype(int)
            new_water_ids  = unclassified_pts[~is_seabed, 4].astype(int)

            self.seabed_ids = np.concatenate([self.seabed_ids, new_seabed_ids])
            self.water_ids  = np.concatenate([self.water_ids,  new_water_ids])

            print(
                f"Reclassified {n_unc:,} unprocessed points "
                f"(Z boundary={z_boundary:.3f}, position='{position}'): "
                f"{len(new_seabed_ids):,} -> seabed, {len(new_water_ids):,} -> water."
            )
        elif n_unc > 0:
            print(f"Warning: {n_unc:,} points remain unclassified (no water reference available).")

        self.assigned_cloud = assign_classes(self.raw_cloud, self.seabed_ids, self.water_ids)
        return self

    # ------------------------------------------------------------------
    # Optional steps
    # ------------------------------------------------------------------

    def build_surface(self, chunk_index: int = -1) -> "SegmentationPipeline":
        """Reconstruct and plot the smooth seabed surface for one chunk.

        ``chunk_index=-1`` aggregates all chunks (can be slow).
        """
        if self.stable_profile is None:
            raise RuntimeError("Segment first.")

        seabed_pts_list = []
        for i, chunk in enumerate(self.chunks):
            if chunk_index != -1 and i != chunk_index:
                continue
            z = chunk["chunk"][:, 2]
            p = self.stable_profile
            if p["position"] == "above":
                mask = z < p["range_min"]
            elif p["position"] == "below":
                mask = z > p["range_max"]
            else:
                mask = (z >= p["range_min"]) & (z <= p["range_max"])
            seabed_pts_list.append(chunk["chunk"][mask, :3])

        if not seabed_pts_list:
            print("No seabed points to reconstruct surface from.")
            return self

        pts = np.vstack(seabed_pts_list)
        X_g, Y_g, Z_g = smooth_surface_from_points(pts)
        surf_dir = os.path.join(self.output_dir, "surface")
        path = plot_surface(X_g, Y_g, Z_g, surf_dir)
        print(f"Surface saved: {path}")
        return self

    def visualize_xz(
        self,
        filename: str = "xz_classification.png",
        max_points: int = 8_000,
    ) -> "SegmentationPipeline":
        """Save an XZ scatter plot of the classification result.

        Blue  = predicted water, sienna = predicted seabed, grey = unclassified.
        Green band = water-surface ±3σ zone derived from the predicted water
        points in the *original* (pre-processing) coordinate space.

        The PNG is written to ``<output_dir>/<filename>``.
        """
        if self.raw_cloud is None or self.seabed_ids is None:
            raise RuntimeError("Call .segment() before .visualize_xz().")

        import os
        src = os.path.basename(getattr(self, "_source_path", ""))
        title = src if src else f"{len(self.seabed_ids):,} seabed / {len(self.water_ids):,} water pts"

        path = plot_xz_classification(
            raw_cloud=self.raw_cloud,
            seabed_ids=self.seabed_ids,
            water_ids=self.water_ids,
            output_dir=self.output_dir,
            filename=filename,
            max_points=max_points,
            title=title,
            stable_profile=self.stable_profile,
            z_bias=getattr(self, "_z_bias", 0.0),
        )
        print(f"XZ plot saved: {path}")
        return self

    def save_las(self) -> "SegmentationPipeline":
        """Write classified point cloud to ``<output_dir>/updated_pointcloud.las``."""
        if self.assigned_cloud is None:
            raise RuntimeError("Segment first.")
        save_to_las(self.assigned_cloud, self.output_dir)
        return self

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def run(self, file_path: str) -> "SegmentationPipeline":
        """Execute the full pipeline (load → preprocess → chunk → GMM → segment → visualize)."""
        self._source_path = file_path
        return (
            self.load(file_path)
            .preprocess()
            .chunk()
            .fit_gmm()
            .segment()
            .visualize_xz()
        )
