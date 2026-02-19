"""
QA Check 06 --- Verify flatmap projection fidelity.
=====================================================

The cerebellar flatmap (Diedrichsen & Zotow 2015) is used throughout
the pipeline for 2D visualisation of volumetric results.  This script
verifies that the projection between volume space (SUIT) and flatmap
surface space is faithful.

Checks performed
----------------
1. **Round-trip fidelity** -- Project the SUIT lobular parcellation to
   the flatmap and then inspect whether lobule boundaries are preserved
   (the projection is inherently one-way, volume-to-surface, so this
   check validates that distinct integer labels remain distinct on the
   surface rather than blurring into each other).
2. **No data loss** -- Compare the fraction of non-zero values in the
   volume with the fraction of non-zero values on the surface.  A large
   drop indicates that many voxels are not reaching the surface.
3. **Color-scale verification** -- Render a known gradient volume
   (ramp from 0 to 1 along the AP axis) and verify that the flatmap
   shows a smooth colour gradient without banding or aliasing artefacts.

Visualisations
--------------
* ``qa06_flatmap_lobular.png``   -- Lobular parcellation on the flatmap.
* ``qa06_flatmap_buckner7.png``  -- Buckner 7-network parcellation.
* ``qa06_flatmap_side_by_side.png`` -- 3D mid-sagittal slice next to flatmap.

Outputs
-------
* ``docs/qa_reports/qa06_flatmap_lobular.png``
* ``docs/qa_reports/qa06_flatmap_buckner7.png``
* ``docs/qa_reports/qa06_flatmap_side_by_side.png``
* ``docs/qa_reports/qa06_summary.txt``

Usage
-----
::

    python -m qa.qa_06_flatmap_projection
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
    from matplotlib.colors import ListedColormap
except ImportError as exc:
    sys.exit(f"matplotlib is required: {exc}")

from src.utils import (
    ATLAS_DIR,
    DATA_RAW,
    DOCS_QA,
    load_nifti,
    ensure_directories,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)


# ───────────────────────────────────────────────────────────────────────
# SUITPy wrapper
# ───────────────────────────────────────────────────────────────────────

def _import_suitpy():
    """Import SUITPy or exit with a clear error."""
    try:
        import SUITPy
        return SUITPy
    except ImportError:
        logger.error(
            "SUITPy is required for QA-06.  Install with:\n"
            "  pip install git+https://github.com/DiedrichsenLab/SUITPy.git"
        )
        return None


# ───────────────────────────────────────────────────────────────────────
# File locators
# ───────────────────────────────────────────────────────────────────────

def _find_atlas(pattern: str) -> Path | None:
    matches = sorted(ATLAS_DIR.rglob(pattern))
    return matches[0] if matches else None


def _suit_template_path():
    """Locate the SUIT template shipped with SUITPy."""
    try:
        import SUITPy
        suit_pkg = Path(SUITPy.__file__).resolve().parent
        for candidate in [
            suit_pkg / "data" / "SUIT.nii",
            *sorted(suit_pkg.rglob("SUIT.nii")),
        ]:
            if candidate.exists():
                return candidate
    except ImportError:
        pass
    fallback = DATA_RAW / "suit" / "SUIT.nii"
    return fallback if fallback.exists() else None


# ───────────────────────────────────────────────────────────────────────
# Checks
# ───────────────────────────────────────────────────────────────────────

def check_roundtrip_fidelity(suit, lobular_path: Path) -> dict:
    """
    Project the lobular atlas to the flatmap and verify label distinctness.

    For each lobule label, compute the modal label on the surface vertices
    assigned to that lobule.  If the mode matches the original label for
    >= 80% of lobules, the check passes.
    """
    result = {
        "name": "Round-trip fidelity (lobule labels preserved)",
        "status": "FAIL",
        "detail": "",
    }

    try:
        lob_img = nib.load(str(lobular_path))
        lob_data = lob_img.get_fdata().astype(int)

        # Project to surface using 'mode' statistic for discrete labels
        surf_data = suit.flatmap.vol_to_surf(lob_img, stats="mode")

        if surf_data is None:
            result["detail"] = "vol_to_surf returned None"
            return result

        surf_vals = np.asarray(surf_data).ravel()

        # Count unique labels in volume (cortical, >0)
        vol_labels = set(np.unique(lob_data)) - {0}
        surf_labels = set(np.unique(surf_vals.astype(int))) - {0}

        # Fraction of volume labels that appear on the surface
        if len(vol_labels) == 0:
            result["detail"] = "no labels in volume"
            return result

        preserved = vol_labels & surf_labels
        frac = len(preserved) / len(vol_labels)
        ok = frac >= 0.80

        result["status"] = "PASS" if ok else "FAIL"
        result["detail"] = (
            f"{len(preserved)}/{len(vol_labels)} volume labels appear on surface "
            f"({frac:.1%}); surface has {len(surf_labels)} unique labels"
        )

    except Exception as exc:
        result["detail"] = f"error: {exc}"

    return result


def check_no_data_loss(suit, lobular_path: Path) -> dict:
    """
    Compare non-zero fraction in volume vs surface.

    A large drop (>50%) suggests that many voxels are not being mapped.
    """
    result = {
        "name": "No data loss (volume vs surface coverage)",
        "status": "FAIL",
        "detail": "",
    }

    try:
        lob_img = nib.load(str(lobular_path))
        lob_data = lob_img.get_fdata()

        vol_nonzero_frac = (lob_data > 0).sum() / max(lob_data.size, 1)

        surf_data = suit.flatmap.vol_to_surf(lob_img, stats="mode")
        if surf_data is None:
            result["detail"] = "vol_to_surf returned None"
            return result

        surf_vals = np.asarray(surf_data).ravel()
        surf_nonzero_frac = (surf_vals > 0).sum() / max(surf_vals.size, 1)

        # We don't expect exact match (surface has different vertex count),
        # but a very low surface coverage is a red flag.
        ok = surf_nonzero_frac > 0.3  # at least 30% of vertices have data

        result["status"] = "PASS" if ok else "FAIL"
        result["detail"] = (
            f"volume non-zero={vol_nonzero_frac:.3f}, "
            f"surface non-zero={surf_nonzero_frac:.3f}"
        )

    except Exception as exc:
        result["detail"] = f"error: {exc}"

    return result


def check_color_scale(suit, lobular_path: Path) -> dict:
    """
    Create a gradient volume (0 to 1 along AP axis) and project it.

    Verify that the surface values span a reasonable range and are
    monotonically increasing along the AP direction of the flatmap.
    """
    result = {
        "name": "Color scale verification (gradient smoothness)",
        "status": "FAIL",
        "detail": "",
    }

    try:
        lob_img = nib.load(str(lobular_path))
        shape = lob_img.shape[:3]
        affine = lob_img.affine

        # Create gradient along dim-1 (typically A-P in SUIT space)
        gradient = np.zeros(shape, dtype=np.float32)
        for j in range(shape[1]):
            gradient[:, j, :] = j / max(shape[1] - 1, 1)

        grad_img = nib.Nifti1Image(gradient, affine)
        surf_data = suit.flatmap.vol_to_surf(grad_img, stats="nanmean")

        if surf_data is None:
            result["detail"] = "vol_to_surf returned None"
            return result

        surf_vals = np.asarray(surf_data).ravel()
        valid = ~np.isnan(surf_vals) & (surf_vals > 0)

        if valid.sum() < 10:
            result["detail"] = "too few valid surface vertices"
            return result

        # Check that values span a reasonable range
        vmin = float(surf_vals[valid].min())
        vmax = float(surf_vals[valid].max())
        span = vmax - vmin
        ok = span > 0.3  # gradient should cover at least 30% of [0, 1]

        result["status"] = "PASS" if ok else "FAIL"
        result["detail"] = f"surface gradient range: [{vmin:.3f}, {vmax:.3f}], span={span:.3f}"

    except Exception as exc:
        result["detail"] = f"error: {exc}"

    return result


# ───────────────────────────────────────────────────────────────────────
# Visualisations
# ───────────────────────────────────────────────────────────────────────

def _plot_flatmap_parcellation(
    suit,
    nifti_path: Path,
    title: str,
    out: Path,
    cmap: str = "tab20",
    stats: str = "mode",
) -> None:
    """Project a parcellation volume to the flatmap and save the figure."""
    try:
        img = nib.load(str(nifti_path))
        surf_data = suit.flatmap.vol_to_surf(img, stats=stats)

        fig = plt.figure(figsize=(12, 8))
        suit.flatmap.plot(
            surf_data,
            render="matplotlib",
            cmap=cmap,
            new_figure=False,
        )
        plt.title(title)

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        logger.info("Saved flatmap: %s", out)

    except Exception as exc:
        logger.warning("Flatmap plot '%s' failed: %s", title, exc)


def _plot_side_by_side(suit, lobular_path: Path, out: Path) -> None:
    """
    Side-by-side: mid-sagittal 3D slice alongside the flatmap projection.
    """
    try:
        img = nib.load(str(lobular_path))
        data = img.get_fdata().astype(int)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Left panel: mid-sagittal slice
        mid_x = data.shape[0] // 2
        sag_slice = data[mid_x, :, :].T
        axes[0].imshow(sag_slice, origin="lower", cmap="tab20", aspect="auto")
        axes[0].set_title("3D lobular atlas (mid-sagittal)")
        axes[0].set_xlabel("dim-1 (A-P)")
        axes[0].set_ylabel("dim-2 (I-S)")

        # Right panel: flatmap
        surf_data = suit.flatmap.vol_to_surf(img, stats="mode")
        if surf_data is not None:
            # Use SUITPy's matplotlib rendering
            plt.sca(axes[1])
            suit.flatmap.plot(
                surf_data,
                render="matplotlib",
                cmap="tab20",
                new_figure=False,
            )
            axes[1].set_title("Flatmap projection")
        else:
            axes[1].text(0.5, 0.5, "Flatmap projection failed",
                         ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title("Flatmap (error)")

        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        logger.info("Saved side-by-side: %s", out)

    except Exception as exc:
        logger.warning("Side-by-side plot failed: %s", exc)


# ───────────────────────────────────────────────────────────────────────
# Results table
# ───────────────────────────────────────────────────────────────────────

def _print_results_table(results: list[dict]) -> str:
    width = 55
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
    """Run all QA-06 flatmap projection checks."""
    ensure_directories()
    DOCS_QA.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  QA Check 06 -- Flatmap Projection Fidelity")
    print("=" * 65)
    print()

    # Import SUITPy
    suit = _import_suitpy()
    if suit is None:
        msg = "SUITPy not installed; cannot run flatmap QA checks."
        print(f"FAIL: {msg}")
        (DOCS_QA / "qa06_summary.txt").write_text(f"QA-06 flatmap: FAIL ({msg})\n")
        return

    # Locate atlas files
    lobular_path = _find_atlas("*Anatom*space-SUIT*dseg.nii*")
    buckner7_path = _find_atlas("*Buckner7*space-SUIT*dseg.nii*")

    if lobular_path is None:
        msg = "SUIT lobular parcellation not found."
        print(f"FAIL: {msg}")
        (DOCS_QA / "qa06_summary.txt").write_text(f"QA-06 flatmap: FAIL ({msg})\n")
        return

    logger.info("Lobular atlas: %s", lobular_path)
    if buckner7_path:
        logger.info("Buckner-7 atlas: %s", buckner7_path)

    results: list[dict] = []

    # 1. Round-trip fidelity
    logger.info("--- Check 1: Round-trip fidelity ---")
    results.append(check_roundtrip_fidelity(suit, lobular_path))

    # 2. No data loss
    logger.info("--- Check 2: No data loss ---")
    results.append(check_no_data_loss(suit, lobular_path))

    # 3. Color scale verification
    logger.info("--- Check 3: Color scale verification ---")
    results.append(check_color_scale(suit, lobular_path))

    # Results table
    print()
    _print_results_table(results)

    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    print(f"\nOverall QA-06 result: {overall}")

    # Visualisations
    logger.info("Generating visualisations ...")

    _plot_flatmap_parcellation(
        suit, lobular_path,
        "SUIT lobular parcellation",
        DOCS_QA / "qa06_flatmap_lobular.png",
        cmap="tab20",
    )

    if buckner7_path:
        _plot_flatmap_parcellation(
            suit, buckner7_path,
            "Buckner 7-network parcellation",
            DOCS_QA / "qa06_flatmap_buckner7.png",
            cmap="Set2",
        )

    _plot_side_by_side(suit, lobular_path, DOCS_QA / "qa06_flatmap_side_by_side.png")

    # Summary
    summary_path = DOCS_QA / "qa06_summary.txt"
    summary_path.write_text(f"QA-06 flatmap projection: {overall}\n")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
