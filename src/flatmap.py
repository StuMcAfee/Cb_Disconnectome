"""
Cerebellar flatmap visualization using the SUIT toolbox.

Projects disruption results onto the SUIT cerebellar flatmap for
publication-quality figures. Supports two input modes:

1. **Vertex data** (preferred): A 1D array of 28,935 values indexed by SUIT
   surface vertex.  Passed directly to suit.flatmap.plot() with no projection
   step needed.
2. **Volume data** (legacy): A 3D NIfTI volume in SUIT space.  Projected onto
   the flatmap surface via suit.flatmap.vol_to_surf().

Provides functions for:
- Plotting disruption probability maps with custom colormaps
- Plotting parcellation and nuclear target maps
- Generating multi-panel reports for individual lesions
- Comparing disruption results across inference methods

Requires:
    - SUITPy (https://github.com/DiedrichsenLab/SUITPy)
    - matplotlib
    - numpy
    - nibabel

Usage:
    python -m src.flatmap <disruption_volume.nii.gz>
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import nibabel as nib

import SUITPy as suit

from src.utils import load_metadata, DATA_FINAL, DOCS_QA, load_nifti
from src.inference import infer_disruption

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom colormaps
# ---------------------------------------------------------------------------

# Disruption colormap: white -> yellow -> orange -> red -> dark red
_DISRUPTION_COLORS = ["#FFFFFF", "#FFEDA0", "#FEB24C", "#F03B20", "#BD0026"]
DISRUPTION_CMAP = LinearSegmentedColormap.from_list(
    "disruption", _DISRUPTION_COLORS, N=256
)

# Nuclear target colormap: background, fastigial, interposed, dentate
_NUCLEAR_COLORS = ["#333333", "#E31A1C", "#33A02C", "#1F78B4"]
NUCLEAR_CMAP = ListedColormap(_NUCLEAR_COLORS, name="nuclear_targets")

# Inference method display names (order matters for the 2x2 grid)
INFERENCE_METHODS = ["max", "mean", "weighted_sum", "threshold_fraction"]
INFERENCE_LABELS = {
    "max": "Max (default)",
    "mean": "Mean",
    "weighted_sum": "Weighted Sum",
    "threshold_fraction": "Threshold Fraction",
}


# ---------------------------------------------------------------------------
# 1. project_to_flatmap
# ---------------------------------------------------------------------------

def project_to_flatmap(data):
    """
    Get flatmap surface data from either a volume path or vertex array.

    Parameters
    ----------
    data : str, Path, or np.ndarray
        If a string/Path: path to a NIfTI file in SUIT space, projected
        via suit.flatmap.vol_to_surf().
        If an ndarray of shape (N_vertices,): used directly (already in
        surface vertex space).

    Returns
    -------
    surf_data : np.ndarray
        1D array of surface values (one per flatmap vertex).
    """
    if isinstance(data, np.ndarray):
        return data
    volume_path = str(data)
    surf_data = suit.flatmap.vol_to_surf(volume_path)
    return surf_data


# ---------------------------------------------------------------------------
# 2. plot_disruption_flatmap
# ---------------------------------------------------------------------------

def plot_disruption_flatmap(
    disruption_data,
    lesion_data=None,
    output_path=None,
    title="Cortical Disruption Map",
):
    """
    Plot a disruption probability map on the cerebellar flatmap.

    Parameters
    ----------
    disruption_data : str, Path, or np.ndarray
        Disruption data — either a NIfTI path (volume) or a 1D vertex array.
    lesion_data : str, Path, np.ndarray, or None
        Lesion overlay — NIfTI path, vertex array, or None.
    output_path : str or Path, optional
        If provided, the figure is saved to this path.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    surf_data = project_to_flatmap(disruption_data)

    fig = plt.figure(figsize=(10, 8))

    suit.flatmap.plot(
        surf_data,
        render="matplotlib",
        cmap=DISRUPTION_CMAP,
        cscale=[0, 1],
        new_figure=False,
    )

    # Overlay lesion boundary if provided
    if lesion_data is not None:
        lesion_surf = project_to_flatmap(lesion_data)
        lesion_binary = (lesion_surf > 0.1).astype(float)
        suit.flatmap.plot(
            lesion_binary,
            render="matplotlib",
            cmap=ListedColormap(["none", "black"]),
            cscale=[0.5, 1.5],
            new_figure=False,
        )

    plt.title(title, fontsize=14, fontweight="bold")

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=DISRUPTION_CMAP,
        norm=mcolors.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label("Disruption Probability", fontsize=11)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
        logger.info("Saved disruption flatmap: %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# 3. plot_parcellation_flatmap
# ---------------------------------------------------------------------------

def plot_parcellation_flatmap(
    parcellation_volume_path,
    output_path=None,
    title="Cerebellar Parcellation",
):
    """
    Plot a cerebellar parcellation on the flatmap.

    Parameters
    ----------
    parcellation_volume_path : str or Path
        Path to the parcellation NIfTI volume (SUIT space, integer labels).
    output_path : str or Path, optional
        If provided, the figure is saved to this path.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    surf_data = project_to_flatmap(parcellation_volume_path)

    unique_labels = np.unique(surf_data[surf_data > 0])
    if len(unique_labels) > 0:
        vmin = float(unique_labels.min())
        vmax = float(unique_labels.max())
    else:
        vmin, vmax = 0.0, 1.0

    fig = plt.figure(figsize=(10, 8))

    suit.flatmap.plot(
        surf_data,
        render="matplotlib",
        cmap="tab20",
        cscale=[vmin, vmax],
        new_figure=False,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
        logger.info("Saved parcellation flatmap: %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# 4. plot_nuclear_targets_flatmap
# ---------------------------------------------------------------------------

def plot_nuclear_targets_flatmap(
    winner_volume_path,
    output_path=None,
    title="Nuclear Target Map",
):
    """
    Plot a winner-take-all nuclear target assignment on the cerebellar flatmap.

    Color coding:
      - 0 (background/non-cortical): dark grey
      - 1 (fastigial): red
      - 2 (interposed): green
      - 3 (dentate): blue

    Parameters
    ----------
    winner_volume_path : str or Path
        Path to the winner-take-all nuclear target NIfTI (SUIT space).
    output_path : str or Path, optional
        If provided, the figure is saved to this path.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    surf_data = project_to_flatmap(winner_volume_path)

    surf_labels = np.round(surf_data).astype(int)
    surf_labels_shifted = surf_labels + 1

    nuclear_colors = [
        "#333333",  # 0: non-cortical / background
        "#E31A1C",  # 1: fastigial (red)
        "#33A02C",  # 2: interposed - emboliform (green)
        "#33A02C",  # 3: interposed - globose (green)
        "#1F78B4",  # 4: dentate (blue)
    ]
    cmap = ListedColormap(nuclear_colors, name="nuclear_targets_5")

    fig = plt.figure(figsize=(10, 8))

    suit.flatmap.plot(
        surf_labels_shifted.astype(float),
        render="matplotlib",
        cmap=cmap,
        cscale=[0, 4],
        new_figure=False,
    )

    plt.title(title, fontsize=14, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E31A1C", edgecolor="black", label="Fastigial"),
        Patch(facecolor="#33A02C", edgecolor="black", label="Interposed"),
        Patch(facecolor="#1F78B4", edgecolor="black", label="Dentate"),
    ]
    plt.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
        logger.info("Saved nuclear targets flatmap: %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# 5. plot_comparison_methods
# ---------------------------------------------------------------------------

def plot_comparison_methods(lesion_path, output_path=None):
    """
    Compare disruption maps produced by all four inference methods.

    Runs vertex-level inference with each method and arranges the flatmap
    results in a 2x2 subplot grid.

    Parameters
    ----------
    lesion_path : str or Path
        Path to the binary lesion mask NIfTI (SUIT space).
    output_path : str or Path, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    lesion_path = str(lesion_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        "Disruption Map Comparison Across Inference Methods",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for idx, method in enumerate(INFERENCE_METHODS):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        try:
            # Run vertex-level inference
            disruption = infer_disruption(
                lesion_path=lesion_path,
                method=method,
            )

            # Plot directly (disruption is already vertex-indexed)
            plt.sca(ax)
            suit.flatmap.plot(
                disruption,
                render="matplotlib",
                cmap=DISRUPTION_CMAP,
                cscale=[0, 1],
                new_figure=False,
            )

            ax.set_title(
                INFERENCE_LABELS.get(method, method),
                fontsize=13,
                fontweight="bold",
            )

        except Exception as e:
            logger.warning(
                "Failed to run inference with method '%s': %s", method, e
            )
            ax.text(
                0.5, 0.5,
                f"{method}\n(failed)",
                ha="center", va="center",
                fontsize=12, color="red",
                transform=ax.transAxes,
            )
            ax.set_facecolor("#f0f0f0")

    # Add a shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=DISRUPTION_CMAP,
        norm=mcolors.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("Disruption Probability", fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
        logger.info("Saved method comparison: %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point: generate a disruption flatmap."""
    parser = argparse.ArgumentParser(
        description="Project disruption results onto the SUIT cerebellar flatmap"
    )
    parser.add_argument(
        "disruption_input",
        type=str,
        help="Path to disruption volume (SUIT NIfTI) or vertex .npz file",
    )
    parser.add_argument(
        "--lesion", type=str, default=None,
        help="Path to binary lesion mask to overlay on the flatmap",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for the saved figure (e.g. flatmap.png)",
    )
    parser.add_argument(
        "--title", type=str, default="Cortical Disruption Map",
        help="Title for the flatmap figure",
    )
    parser.add_argument(
        "--method", type=str, default="max",
        help="If input is a .npz, which method key to use (default: max)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Determine input type
    input_path = Path(args.disruption_input)
    if input_path.suffix == ".npz":
        npz = np.load(str(input_path))
        key = f"disruption_{args.method}"
        if key not in npz:
            available = [k for k in npz.files if k.startswith("disruption_")]
            raise ValueError(
                f"Key '{key}' not in {input_path}. Available: {available}"
            )
        disruption_data = npz[key]
    else:
        disruption_data = str(input_path)

    # Default output path
    output_path = args.output
    if output_path is None:
        output_path = str(
            input_path.parent / (input_path.stem.replace(".nii", "") + "_flatmap.png")
        )
        logger.info("No output path specified; saving to: %s", output_path)

    fig = plot_disruption_flatmap(
        disruption_data=disruption_data,
        lesion_data=args.lesion,
        output_path=output_path,
        title=args.title,
    )
    plt.close(fig)
    logger.info("Done. Figure saved to: %s", output_path)


if __name__ == "__main__":
    main()
