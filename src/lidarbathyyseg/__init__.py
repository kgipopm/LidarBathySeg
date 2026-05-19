
from .chunking import chunk_pointcloud, select_optimal_chunk_count
from .classification import train_svm_classifier
from .gmm import (
    analyze_gmm_fragments,
    check_best_gmm_model,
    find_best_gmm,
    get_universal_stable_profile,
)
from .io import save_results_to_original_las, save_to_las
from .pipeline import SegmentationPipeline
from .preprocessing import (
    extend_xy_with_nearest_z,
    filter_pointcloud,
    load_pointcloud,
    remove_bias,
    rotate_to_principal_axis,
    sample_xy,
    temporary_plot,
)
from .segmentation import assign_classes, filter_points
from .surface import plot_surface, smooth_surface_from_points

# Backward-compatible aliases (old function names)
from .classification import prepare_and_vizualize_decission_borders
from .segmentation import filter_and_visualize_numpy_points
from .io import save_results_to_original_las as save_results_to_original_pointcloud

__all__ = [
    # preprocessing
    "load_pointcloud", "filter_pointcloud", "rotate_to_principal_axis",
    "remove_bias", "temporary_plot", "extend_xy_with_nearest_z", "sample_xy",
    # chunking
    "chunk_pointcloud", "select_optimal_chunk_count",
    # gmm
    "check_best_gmm_model", "find_best_gmm", "analyze_gmm_fragments",
    "get_universal_stable_profile",
    # segmentation
    "filter_points", "assign_classes",
    # surface
    "smooth_surface_from_points", "plot_surface",
    # classification
    "train_svm_classifier",
    # io
    "save_to_las", "save_results_to_original_las",
    # pipeline
    "SegmentationPipeline",
    # back-compat aliases
    "filter_and_visualize_numpy_points",
    "prepare_and_vizualize_decission_borders",
    "save_results_to_original_pointcloud",
]
