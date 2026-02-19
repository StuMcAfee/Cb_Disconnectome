"""
QA Check 02 --- Validate the cortico-nuclear probability map.
==============================================================

The cortico-nuclear probability map (``data/interim/cortico_nuclear_prob_map.nii.gz``)
assigns each cerebellar cortex voxel a probability distribution over the four
deep cerebellar nuclei (fastigial, emboliform, globose, dentate).

This script verifies the map against a battery of quantitative and anatomical
plausibility checks.

Checks performed
----------------
1. **Coverage** -- Every cortical voxel (defined by the SUIT lobular atlas)
   must have at least one non-zero nucleus probability.
2. **Normalization** -- Per-voxel probabilities must sum to 1.0 within a
   configurable tolerance (default ``atol=1e-5``).
3. **Non-cortex exclusion** -- All voxels outside the cortical mask must
   have zero probability for every nucleus.
4. **Anatomical plausibility** --
   - Midline (|x| < 5 mm): fastigial should be the dominant nucleus.
   - Lateral (|x| > 15 mm): dentate should be the dominant nucleus.
5. **Visualisations** --
   - Three-plane montage (axial, coronal, sagittal) of the winner-take-all map.
   - Flatmap projection of the dominant nucleus (via SUITPy).
   - Histogram of per-voxel max probability.
   - Scatter plot of |x|-coordinate vs. dentate probability.

Outputs
-------
* ``docs/qa_reports/qa02_montage.png``
* ``docs/qa_reports/qa02_flatmap.png``
* ``docs/qa_reports/qa02_histogram.png``
* ``docs/qa_reports/qa02_scatter_x_vs_prob.png``
* ``docs/qa_reports/qa02_summary.txt``

Usage
-----
::

    python -m qa.qa_02_cortico_nuclear
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    sys.exit(f"matplotlib is required: {exc}")

from src.utils import (
    ATLAS_DIR,
    DATA_INTERIM,
    DOCS_QA,
    load_nifti,
    get_mm_coordinate_grid,
    ensure_directories,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

NUCLEUS_NAMES = ["fastigial", "emboliform", "globose", "dentate"]
NORM_ATOL = 1e-5

# ───────────────────────────────────────────────────────────────────────
# File locators
# ───────────────────────────────────────────────────────────────────────

def _find_prob_map() -> Path | None:
    """Locate the cortico-nuclear probability map."""
    p = DATA_INTERIM / "cortico_nuclear_prob_map.nii.gz"
    if p.exists():
        return p
    # Broader search
    candidates = sorted(DATA_INTERIM.rglob("*cortico_nuclear*prob*.nii*"))
    return candidates[0] if candidates else None


def _find_lobular_atlas() -> Path | None:
    """Locate the SUIT lobular parcellation."""
    candidates = sorted(ATLAS_DIR.rglob("*Anatom*space-SUIT*dseg.nii*"))
    return candidates[0] if candidates else None


# ───────────────────────────────────────────────────────────────────────
# Checks
# ───────────────────────────────────────────────────────────────────────

def check_coverage(prob_data: np.ndarray, cortex_mask: np.ndarray) -> dict:
    """Every cortical voxel should have non-zero total probability."""
    total = prob_data.sum(axis=-1)
    cortex_nonzero = total[cortex_mask] > 0
    n_total = cortex_mask.sum()
    n_nonzero = cortex_nonzero.sum()
    fraction = n_nonzero / max(n_total, 1)
    ok = fraction >= 1.0 - 1e-6  # allow floating-point rounding

    return {
        "name": "Coverage (cortical voxels with nonzero prob)",
        "status": "PASS" if ok else "FAIL",
        "detail": f"{n_nonzero}/{n_total} cortical voxels covered ({fraction:.6f})",
    }


def check_normalization(prob_data: np.ndarray, cortex_mask: np.ndarray) -> dict:
    """Per-voxel probabilities should sum to 1.0 within tolerance."""
    total = prob_data.sum(axis=-1)
    cortex_sums = total[cortex_mask]
    deviations = np.abs(cortex_sums - 1.0)
    max_dev = deviations.max() if deviations.size > 0 else 0.0
    ok = max_dev <= NORM_ATOL

    return {
        "name": "Normalization (probabilities sum to 1.0)",
        "status": "PASS" if ok else "FAIL",
        "detail": f"max deviation from 1.0 = {max_dev:.2e} (tol={NORM_ATOL:.0e})",
    }


def check_noncortex_exclusion(prob_data: np.ndarray, cortex_mask: np.ndarray) -> dict:
    """Non-cortex voxels should have all-zero probabilities."""
    non_cortex = ~cortex_mask
    non_cortex_vals = prob_data[non_cortex]
    n_nonzero = (non_cortex_vals.sum(axis=-1) > 0).sum()
    n_total = non_cortex.sum()
    ok = n_nonzero == 0

    return {
        "name": "Non-cortex exclusion (all zeros outside cortex)",
        "status": "PASS" if ok else "FAIL",
        "detail": f"{n_nonzero}/{n_total} non-cortex voxels with nonzero values",
    }


def check_anatomical_plausibility(
    prob_data: np.ndarray,
    cortex_mask: np.ndarray,
    affine: np.ndarray,
) -> dict:
    """
    Midline cortex should project mainly to fastigial; lateral cortex
    should project mainly to dentate.
    """
    shape_3d = prob_data.shape[:3]
    mm_grid = get_mm_coordinate_grid(shape_3d, affine)
    x_mm = mm_grid[..., 0]

    # Midline voxels (|x| < 5 mm) in cortex
    midline_mask = cortex_mask & (np.abs(x_mm) < 5.0)
    # Lateral voxels (|x| > 15 mm) in cortex
    lateral_mask = cortex_mask & (np.abs(x_mm) > 15.0)

    issues = []

    # Midline: fastigial (index 0) should be dominant
    if midline_mask.sum() > 0:
        midline_probs = prob_data[midline_mask]  # (N, 4)
        avg_midline = midline_probs.mean(axis=0)
        dominant_mid = NUCLEUS_NAMES[np.argmax(avg_midline)]
        if dominant_mid != "fastigial":
            issues.append(
                f"midline dominant nucleus is '{dominant_mid}', expected 'fastigial' "
                f"(avg probs: {dict(zip(NUCLEUS_NAMES, avg_midline.round(3)))})"
            )
        logger.info("  Midline avg probs: %s", dict(zip(NUCLEUS_NAMES, avg_midline.round(4))))
    else:
        issues.append("no midline cortical voxels found")

    # Lateral: dentate (index 3) should be dominant
    if lateral_mask.sum() > 0:
        lateral_probs = prob_data[lateral_mask]
        avg_lateral = lateral_probs.mean(axis=0)
        dominant_lat = NUCLEUS_NAMES[np.argmax(avg_lateral)]
        if dominant_lat != "dentate":
            issues.append(
                f"lateral dominant nucleus is '{dominant_lat}', expected 'dentate' "
                f"(avg probs: {dict(zip(NUCLEUS_NAMES, avg_lateral.round(3)))})"
            )
        logger.info("  Lateral avg probs: %s", dict(zip(NUCLEUS_NAMES, avg_lateral.round(4))))
    else:
        issues.append("no lateral cortical voxels found")

    ok = len(issues) == 0
    return {
        "name": "Anatomical plausibility (midline->fastigial, lateral->dentate)",
        "status": "PASS" if ok else "FAIL",
        "detail": "; ".join(issues) if issues else "midline=fastigial, lateral=dentate",
    }


# ───────────────────────────────────────────────────────────────────────
# Visualisations
# ───────────────────────────────────────────────────────────────────────

def _plot_montage(prob_data: np.ndarray, affine: np.ndarray, out: Path) -> None:
    """Three-plane montage of the winner-take-all map."""
    winner = np.argmax(prob_data, axis=-1).astype(np.float32)
    cortex_mask = prob_data.sum(axis=-1) > 0
    winner[~cortex_mask] = np.nan

    winner_img = nib.Nifti1Image(winner, affine)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mode, label in zip(
        axes, ["x", "y", "z"], ["Sagittal", "Coronal", "Axial"]
    ):
        try:
            from nilearn import plotting as ni_plot
            display = ni_plot.plot_stat_map(
                winner_img,
                display_mode=mode,
                cut_coords=3,
                axes=ax,
                title=f"Winner-take-all ({label})",
                colorbar=True,
                cmap="Set1",
            )
        except Exception:
            ax.set_title(f"{label} (plot error)")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    logger.info("Saved montage: %s", out)


def _plot_flatmap(prob_data: np.ndarray, affine: np.ndarray, out: Path) -> None:
    """Flatmap projection of the dominant nucleus using SUITPy."""
    try:
        import SUITPy as suit

        winner = np.argmax(prob_data, axis=-1).astype(np.float32) + 1
        cortex_mask = prob_data.sum(axis=-1) > 0
        winner[~cortex_mask] = 0

        winner_img = nib.Nifti1Image(winner, affine)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        # SUITPy flatmap projection
        flatmap_data = suit.flatmap.vol_to_surf(
            winner_img,
            stats="mode",
        )
        suit.flatmap.plot(
            flatmap_data,
            render="matplotlib",
            cmap="Set1",
            new_figure=False,
        )
        ax.set_title("Cortico-nuclear winner -- flatmap")

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        logger.info("Saved flatmap: %s", out)

    except ImportError:
        logger.warning("SUITPy not available -- skipping flatmap visualisation")
    except Exception as exc:
        logger.warning("Flatmap generation failed: %s", exc)


def _plot_histogram(prob_data: np.ndarray, cortex_mask: np.ndarray, out: Path) -> None:
    """Histogram of per-voxel maximum probability."""
    max_probs = prob_data[cortex_mask].max(axis=-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_probs, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Max nucleus probability per cortical voxel")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of dominant-nucleus probability")
    ax.axvline(0.5, color="red", linestyle="--", label="p = 0.5")
    ax.legend()

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    logger.info("Saved histogram: %s", out)


def _plot_scatter_x_vs_prob(
    prob_data: np.ndarray,
    cortex_mask: np.ndarray,
    affine: np.ndarray,
    out: Path,
) -> None:
    """Scatter of |x|-coordinate vs fastigial / dentate probability."""
    shape_3d = prob_data.shape[:3]
    mm_grid = get_mm_coordinate_grid(shape_3d, affine)
    x_mm = mm_grid[..., 0]

    x_vals = np.abs(x_mm[cortex_mask])
    fastigial_p = prob_data[cortex_mask][:, 0]
    dentate_p = prob_data[cortex_mask][:, 3]

    # Subsample for readability (max 5 000 points)
    n = len(x_vals)
    if n > 5000:
        idx = np.random.default_rng(42).choice(n, 5000, replace=False)
        x_vals = x_vals[idx]
        fastigial_p = fastigial_p[idx]
        dentate_p = dentate_p[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(x_vals, fastigial_p, s=2, alpha=0.3, c="steelblue")
    axes[0].set_xlabel("|x| (mm)")
    axes[0].set_ylabel("P(fastigial)")
    axes[0].set_title("Fastigial probability vs |x|")

    axes[1].scatter(x_vals, dentate_p, s=2, alpha=0.3, c="firebrick")
    axes[1].set_xlabel("|x| (mm)")
    axes[1].set_ylabel("P(dentate)")
    axes[1].set_title("Dentate probability vs |x|")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    logger.info("Saved scatter plot: %s", out)


# ───────────────────────────────────────────────────────────────────────
# Results table
# ───────────────────────────────────────────────────────────────────────

def _print_results_table(results: list[dict]) -> str:
    width = 60
    sep = "+" + "-" * width + "+" + "-" * 8 + "+"
    header = f"| {'Check':<{width - 1}} | {'Result':>6} |"

    lines = [sep, header, sep]
    for r in results:
        name = r.get("name", "?")[:width - 1]
        status = r.get("status", "N/A")
        lines.append(f"| {name:<{width - 1}} | {status:>6} |")
    lines.append(sep)

    table = "\n".join(lines)
    print(table)
    for r in results:
        print(f"  Detail: {r.get('detail', '')}")
    return table


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all QA-02 checks on the cortico-nuclear probability map."""
    ensure_directories()
    DOCS_QA.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  QA Check 02 -- Cortico-Nuclear Probability Map")
    print("=" * 65)
    print()

    # Load probability map
    prob_path = _find_prob_map()
    if prob_path is None:
        print("FAIL: Cortico-nuclear probability map not found.")
        print("      Run  python -m src.cortico_nuclear_map  first.")
        (DOCS_QA / "qa02_summary.txt").write_text("QA-02 cortico-nuclear map: FAIL (file not found)\n")
        return

    logger.info("Loading probability map: %s", prob_path)
    prob_data, affine, _ = load_nifti(prob_path)
    assert prob_data.ndim == 4 and prob_data.shape[-1] == 4, (
        f"Expected 4D map with 4 nuclei, got shape {prob_data.shape}"
    )

    # Load lobular atlas for cortex mask
    lob_path = _find_lobular_atlas()
    if lob_path is None:
        print("FAIL: Lobular parcellation not found.")
        (DOCS_QA / "qa02_summary.txt").write_text("QA-02 cortico-nuclear map: FAIL (atlas not found)\n")
        return

    lob_data, _, _ = load_nifti(lob_path, dtype=int)

    # Cortical labels: exclude deep nuclei labels (typically >=29)
    # The cortico_nuclear_map module already zeroes non-cortex voxels,
    # but we replicate the mask from the atlas for an independent check.
    # Labels 1-28 are cortical lobules; 29+ are nuclei in the default map.
    cortex_labels = set(range(1, 29))
    cortex_mask = np.isin(lob_data, list(cortex_labels))

    results: list[dict] = []

    # 1. Coverage
    results.append(check_coverage(prob_data, cortex_mask))

    # 2. Normalization
    results.append(check_normalization(prob_data, cortex_mask))

    # 3. Non-cortex exclusion
    results.append(check_noncortex_exclusion(prob_data, cortex_mask))

    # 4. Anatomical plausibility
    results.append(check_anatomical_plausibility(prob_data, cortex_mask, affine))

    # Results table
    print()
    _print_results_table(results)

    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    print(f"\nOverall QA-02 result: {overall}")

    # Visualisations
    logger.info("Generating visualisations ...")
    _plot_montage(prob_data, affine, DOCS_QA / "qa02_montage.png")
    _plot_flatmap(prob_data, affine, DOCS_QA / "qa02_flatmap.png")
    _plot_histogram(prob_data, cortex_mask, DOCS_QA / "qa02_histogram.png")
    _plot_scatter_x_vs_prob(prob_data, cortex_mask, affine, DOCS_QA / "qa02_scatter_x_vs_prob.png")

    # Summary
    summary_path = DOCS_QA / "qa02_summary.txt"
    summary_path.write_text(f"QA-02 cortico-nuclear map: {overall}\n")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
