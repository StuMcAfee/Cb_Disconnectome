"""
Cerebellar flatmap visualization using the SUIT toolbox.

Projects disruption results onto the SUIT cerebellar flatmap for
publication-quality figures. Provides functions for:

- Projecting 3D SUIT-space volumes onto the cerebellar flatmap surface
- Plotting disruption probability maps with custom colormaps
- Plotting parcellation and nuclear target maps
- Generating comprehensive multi-panel reports for individual lesions
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
from src.inference import infer_disruption, disruption_to_volume

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

def project_to_flatmap(volume_path):
    """
    Project a 3D volume in SUIT space onto the cerebellar flatmap surface.

    Uses SUITPy's vol_to_surf function to map voxel values from a NIfTI
    volume (in SUIT space) onto the flatmap surface representation.

    Parameters
    ----------
    volume_path : str or Path
        Path to a NIfTI file in SUIT space.

    Returns
    -------
    surf_data : np.ndarray
        1D array of surface values (one per flatmap vertex).
    """
    volume_path = str(volume_path)
    surf_data = suit.flatmap.vol_to_surf(volume_path)
    return surf_data


# ---------------------------------------------------------------------------
# 2. plot_disruption_flatmap
# ---------------------------------------------------------------------------

def plot_disruption_flatmap(
    disruption_volume_path,
    lesion_volume_path=None,
    output_path=None,
    title="Cortical Disruption Map",
):
    """
    Plot a disruption probability map on the cerebellar flatmap.

    Uses a custom white-to-dark-red colormap to show the probability of
    cortical disruption for each region of the cerebellar cortex.

    Parameters
    ----------
    disruption_volume_path : str or Path
        Path to the disruption probability volume (SUIT space, values 0-1).
    lesion_volume_path : str or Path, optional
        Path to the binary lesion mask volume. If provided, the lesion
        boundary is overlaid on the flatmap.
    output_path : str or Path, optional
        If provided, the figure is saved to this path.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    # Project disruption volume to flatmap surface
    surf_data = project_to_flatmap(disruption_volume_path)

    # Create the figure
    fig = plt.figure(figsize=(10, 8))

    # Plot the flatmap with the disruption colormap
    suit.flatmap.plot(
        surf_data,
        render="matplotlib",
        cmap=DISRUPTION_CMAP,
        cscale=[0, 1],
        new_figure=False,
    )

    # Overlay lesion boundary if provided
    if lesion_volume_path is not None:
        lesion_surf = project_to_flatmap(lesion_volume_path)
        # Threshold the projected lesion to create a binary mask, then
        # overlay the boundary with a dark outline
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

    # Save if requested
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

    Uses a categorical colormap (tab20) suitable for discrete parcel labels.

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
        The generated figure.
    """
    # Project parcellation to flatmap surface
    surf_data = project_to_flatmap(parcellation_volume_path)

    # Determine the range of non-zero labels for color scaling
    unique_labels = np.unique(surf_data[surf_data > 0])
    if len(unique_labels) > 0:
        vmin = float(unique_labels.min())
        vmax = float(unique_labels.max())
    else:
        vmin, vmax = 0.0, 1.0

    # Create the figure
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

    The input volume should contain integer labels from the cortico-nuclear
    winner-take-all map (see src.cortico_nuclear_map).

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
        The generated figure.
    """
    # Project the winner-take-all volume to flatmap
    surf_data = project_to_flatmap(winner_volume_path)

    # The winner map uses 0=fastigial, 1=emboliform, 2=globose, 3=dentate
    # and -1 for non-cortical.  Shift so that non-cortical=-1 maps to index 0
    # and nuclear targets map to indices 1-4.
    # Round to nearest integer and shift.
    surf_labels = np.round(surf_data).astype(int)
    surf_labels_shifted = surf_labels + 1  # now: 0=non-cortical, 1=fast, 2=emb, 3=glob, 4=dent

    # Build a colormap with 5 entries:
    # 0=background, 1=fastigial(red), 2=emboliform(red-orange), 3=globose(green), 4=dentate(blue)
    # Combine emboliform and globose as "interposed" visually
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

    # Add legend
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
# 5. generate_full_report
# ---------------------------------------------------------------------------

def generate_full_report(lesion_path, disruption, metadata_path, output_dir):
    """
    Generate a complete visualization report for a single lesion case.

    Produces four output figures:
      (a) Flatmap of disruption probabilities
      (b) Bar chart of per-parcel disruption (sorted, top 20 parcels)
      (c) Summary table of top 10 disrupted parcels with names and values
      (d) Comparison of disruption maps across all 4 inference methods

    Parameters
    ----------
    lesion_path : str or Path
        Path to the binary lesion mask NIfTI (SUIT space).
    disruption : dict
        Disruption results dictionary mapping parcel names/IDs to
        disruption probability values (as returned by infer_disruption).
    metadata_path : str or Path
        Path to the JSON metadata file containing parcel name mappings
        and model configuration.
    output_dir : str or Path
        Directory where output figures will be saved.

    Returns
    -------
    dict
        Mapping of figure names to their file paths.
    """
    lesion_path = Path(lesion_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    saved_figures = {}

    # Derive a case identifier from the lesion filename
    case_id = lesion_path.stem.replace(".nii", "").replace(".gz", "")

    # --- (a) Disruption flatmap ---
    logger.info("Generating disruption flatmap for %s...", case_id)
    disruption_vol_path = output_dir / f"{case_id}_disruption.nii.gz"
    disruption_to_volume(disruption, str(disruption_vol_path))
    fig_flatmap = plot_disruption_flatmap(
        disruption_volume_path=str(disruption_vol_path),
        lesion_volume_path=str(lesion_path),
        output_path=str(output_dir / f"{case_id}_flatmap.png"),
        title=f"Cortical Disruption: {case_id}",
    )
    plt.close(fig_flatmap)
    saved_figures["flatmap"] = output_dir / f"{case_id}_flatmap.png"

    # --- (b) Bar chart of per-parcel disruption (top 20) ---
    logger.info("Generating per-parcel bar chart for %s...", case_id)

    # Sort parcels by disruption probability in descending order
    sorted_parcels = sorted(
        disruption.items(), key=lambda x: x[1], reverse=True
    )
    top_20 = sorted_parcels[:20]

    if top_20:
        parcel_names = [str(p[0]) for p in top_20]
        parcel_values = [float(p[1]) for p in top_20]

        fig_bar, ax_bar = plt.subplots(figsize=(10, 7))
        bars = ax_bar.barh(
            range(len(parcel_names)),
            parcel_values,
            color=[DISRUPTION_CMAP(v) for v in parcel_values],
            edgecolor="grey",
            linewidth=0.5,
        )
        ax_bar.set_yticks(range(len(parcel_names)))
        ax_bar.set_yticklabels(parcel_names, fontsize=9)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Disruption Probability", fontsize=11)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_title(
            f"Top {len(top_20)} Disrupted Parcels: {case_id}",
            fontsize=13,
            fontweight="bold",
        )
        ax_bar.axvline(x=0.5, color="grey", linestyle="--", alpha=0.5)
        plt.tight_layout()

        bar_path = output_dir / f"{case_id}_barchart.png"
        fig_bar.savefig(str(bar_path), dpi=200, bbox_inches="tight")
        plt.close(fig_bar)
        saved_figures["barchart"] = bar_path
        logger.info("Saved bar chart: %s", bar_path)

    # --- (c) Summary table (top 10) ---
    logger.info("Generating summary table for %s...", case_id)

    top_10 = sorted_parcels[:10]
    fig_table, ax_table = plt.subplots(figsize=(8, 4))
    ax_table.axis("off")

    if top_10:
        # Build table data
        col_labels = ["Rank", "Parcel", "Disruption Prob."]
        table_data = []
        for rank, (name, value) in enumerate(top_10, start=1):
            table_data.append([str(rank), str(name), f"{float(value):.3f}"])

        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.4)

        # Style header row
        for col_idx in range(len(col_labels)):
            table[0, col_idx].set_facecolor("#4472C4")
            table[0, col_idx].set_text_props(color="white", fontweight="bold")

        # Color-code disruption values
        for row_idx in range(1, len(table_data) + 1):
            val = float(table_data[row_idx - 1][2])
            table[row_idx, 2].set_facecolor(DISRUPTION_CMAP(val))
            # Use white text on dark backgrounds
            if val > 0.6:
                table[row_idx, 2].set_text_props(color="white")

    ax_table.set_title(
        f"Top 10 Disrupted Parcels: {case_id}",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    table_path = output_dir / f"{case_id}_table.png"
    fig_table.savefig(str(table_path), dpi=200, bbox_inches="tight")
    plt.close(fig_table)
    saved_figures["table"] = table_path
    logger.info("Saved summary table: %s", table_path)

    # --- (d) Comparison across inference methods ---
    logger.info("Generating method comparison for %s...", case_id)
    comparison_path = output_dir / f"{case_id}_methods_comparison.png"
    fig_comparison = plot_comparison_methods(
        lesion_path=str(lesion_path),
        occupancy_path=str(metadata.get("occupancy_path", "")),
        output_path=str(comparison_path),
    )
    if fig_comparison is not None:
        plt.close(fig_comparison)
    saved_figures["comparison"] = comparison_path

    logger.info(
        "Full report complete for %s. %d figures saved to %s",
        case_id,
        len(saved_figures),
        output_dir,
    )
    return saved_figures


# ---------------------------------------------------------------------------
# 6. plot_comparison_methods
# ---------------------------------------------------------------------------

def plot_comparison_methods(lesion_path, occupancy_path, output_path=None):
    """
    Compare disruption maps produced by all four inference methods.

    Runs inference with each method (max, mean, weighted_sum,
    threshold_fraction), projects each result onto the flatmap, and
    arranges them in a 2x2 subplot grid.

    Parameters
    ----------
    lesion_path : str or Path
        Path to the binary lesion mask NIfTI (SUIT space).
    occupancy_path : str or Path
        Path to the pathway occupancy map used for inference.
    output_path : str or Path, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The generated figure, or None if inference fails.
    """
    lesion_path = str(lesion_path)
    occupancy_path = str(occupancy_path)

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
            # Run inference with this method
            disruption = infer_disruption(
                lesion_path=lesion_path,
                occupancy_path=occupancy_path,
                method=method,
            )

            # Convert disruption dict to a temporary volume for projection
            tmp_vol_path = Path(output_path).parent / f"_tmp_{method}.nii.gz" if output_path else Path(f"/tmp/_tmp_{method}.nii.gz")
            disruption_to_volume(disruption, str(tmp_vol_path))

            # Project to surface
            surf_data = project_to_flatmap(str(tmp_vol_path))

            # Plot on the corresponding axis
            plt.sca(ax)
            suit.flatmap.plot(
                surf_data,
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

            # Clean up temporary file
            if tmp_vol_path.exists():
                tmp_vol_path.unlink()

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
    """Command-line entry point: generate a disruption flatmap from a volume."""
    parser = argparse.ArgumentParser(
        description="Project a disruption volume onto the SUIT cerebellar flatmap"
    )
    parser.add_argument(
        "disruption_volume",
        type=str,
        help="Path to the disruption probability volume (SUIT space NIfTI)",
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Generate default output path if not specified
    output_path = args.output
    if output_path is None:
        vol_path = Path(args.disruption_volume)
        output_path = str(
            vol_path.parent / (vol_path.stem.replace(".nii", "") + "_flatmap.png")
        )
        logger.info("No output path specified; saving to: %s", output_path)

    fig = plot_disruption_flatmap(
        disruption_volume_path=args.disruption_volume,
        lesion_volume_path=args.lesion,
        output_path=output_path,
        title=args.title,
    )
    plt.close(fig)
    logger.info("Done. Figure saved to: %s", output_path)


if __name__ == "__main__":
    main()
