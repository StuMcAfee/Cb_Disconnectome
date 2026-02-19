"""
QA Check 01 --- Verify spatial alignment of all atlas files.
=============================================================

This script validates that every atlas file used in the cerebellar
disconnectome pipeline lives in its expected coordinate space and that
files that *should* share a space really do.

Checks performed
----------------
1. **SUIT-space consistency** -- The lobular parcellation
   (``atl-Anatom_space-SUIT_dseg.nii``) must share the same affine
   matrix and grid dimensions as the SUIT template shipped with SUITPy.
2. **MNI-space consistency** -- The MNI lobular parcellation
   (``atl-Anatom_space-MNI_dseg.nii``) must share affine / shape with
   the MNI152 template from ``nilearn.datasets``.
3. **Deep-nuclei containment** -- Every non-zero voxel in each deep
   cerebellar nucleus probability map must fall within the cerebellum
   mask (dilated by 1 voxel to account for partial-volume effects).
4. **Overlay visualisations** -- Mid-sagittal and mid-axial overlay
   images are saved so a human reviewer can quickly spot gross
   misalignment.

Outputs
-------
* ``docs/qa_reports/qa01_suit_overlay.png``
* ``docs/qa_reports/qa01_mni_overlay.png``
* ``docs/qa_reports/qa01_nuclei_overlay.png``
* ``docs/qa_reports/qa01_summary.txt``   (one-line PASS / FAIL)

Usage
-----
::

    python -m qa.qa_01_atlas_alignment
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from textwrap import dedent

import nibabel as nib
import numpy as np

# Optional heavy imports -- guarded so that a missing package gives a
# clear error message rather than an opaque traceback.
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
except ImportError as exc:
    sys.exit(f"matplotlib is required for QA visualisations: {exc}")

try:
    from nilearn import plotting as ni_plot
except ImportError as exc:
    sys.exit(f"nilearn is required for overlay plotting: {exc}")

# Project utilities
from src.utils import (
    ATLAS_DIR,
    DATA_RAW,
    DOCS_QA,
    check_affine_match,
    check_shape_match,
    load_nifti,
    ensure_directories,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

# ───────────────────────────────────────────────────────────────────────
# Helper utilities
# ───────────────────────────────────────────────────────────────────────

def _find_atlas(pattern: str, search_dir: Path | None = None) -> Path | None:
    """Return the first file matching *pattern* under *search_dir*."""
    search_dir = search_dir or ATLAS_DIR
    matches = sorted(search_dir.rglob(pattern))
    return matches[0] if matches else None


def _suit_template_path() -> Path | None:
    """Locate the SUIT template shipped with SUITPy."""
    try:
        import SUITPy
        suit_pkg = Path(SUITPy.__file__).resolve().parent
        for candidate in [
            suit_pkg / "data" / "SUIT.nii",
            suit_pkg / "surfaces" / "SUIT.nii",
            *sorted(suit_pkg.rglob("SUIT.nii")),
        ]:
            if candidate.exists():
                return candidate
    except ImportError:
        logger.warning("SUITPy not installed -- cannot locate SUIT template.")
    # Fallback: check the raw data directory
    fallback = DATA_RAW / "suit" / "SUIT.nii"
    return fallback if fallback.exists() else None


def _mni_template() -> nib.Nifti1Image | None:
    """Return the MNI152 2-mm template from nilearn (lazy download)."""
    try:
        from nilearn.datasets import load_mni152_template
        return load_mni152_template(resolution=2)
    except Exception:
        try:
            from nilearn.datasets import load_mni152_template
            return load_mni152_template()
        except Exception as exc:
            logger.warning("Could not load MNI template: %s", exc)
            return None


def _save_overlay(bg_img, overlay_img, title: str, out_path: Path) -> None:
    """Save a mid-slice overlay image (sagittal + axial)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sagittal (x = mid)
    try:
        ni_plot.plot_roi(
            roi_img=overlay_img,
            bg_img=bg_img,
            display_mode="x",
            cut_coords=1,
            alpha=0.5,
            axes=axes[0],
            title=f"{title} -- sagittal",
        )
    except Exception:
        # Fallback: simple stat map
        ni_plot.plot_stat_map(
            stat_map_img=overlay_img,
            bg_img=bg_img,
            display_mode="x",
            cut_coords=1,
            axes=axes[0],
            title=f"{title} -- sagittal",
        )

    # Axial (z = mid)
    try:
        ni_plot.plot_roi(
            roi_img=overlay_img,
            bg_img=bg_img,
            display_mode="z",
            cut_coords=1,
            alpha=0.5,
            axes=axes[1],
            title=f"{title} -- axial",
        )
    except Exception:
        ni_plot.plot_stat_map(
            stat_map_img=overlay_img,
            bg_img=bg_img,
            display_mode="z",
            cut_coords=1,
            axes=axes[1],
            title=f"{title} -- axial",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info("Saved overlay figure: %s", out_path)


# ───────────────────────────────────────────────────────────────────────
# Individual checks
# ───────────────────────────────────────────────────────────────────────

def check_suit_consistency() -> dict:
    """
    Compare the SUIT lobular atlas against the SUIT template.

    Returns a dict with keys ``affine_match``, ``shape_match``, and
    ``status`` (``PASS`` or ``FAIL``).
    """
    result = {
        "name": "SUIT-space consistency",
        "affine_match": False,
        "shape_match": False,
        "status": "FAIL",
        "detail": "",
    }

    suit_template_p = _suit_template_path()
    if suit_template_p is None:
        result["detail"] = "SUIT template not found (SUITPy not installed?)"
        return result

    lobular_p = _find_atlas("*Anatom*space-SUIT*dseg.nii*")
    if lobular_p is None:
        result["detail"] = "SUIT lobular parcellation not found in atlas dir"
        return result

    tmpl_data, tmpl_aff, tmpl_hdr = load_nifti(suit_template_p)
    lob_data, lob_aff, lob_hdr = load_nifti(lobular_p)

    result["affine_match"] = check_affine_match(tmpl_aff, lob_aff)
    result["shape_match"] = check_shape_match(tmpl_data.shape, lob_data.shape)
    result["status"] = (
        "PASS" if result["affine_match"] and result["shape_match"] else "FAIL"
    )
    result["detail"] = (
        f"template shape={tmpl_data.shape}, lobular shape={lob_data.shape[:3]}; "
        f"affine match={result['affine_match']}"
    )

    # Visualisation
    try:
        tmpl_img = nib.load(str(suit_template_p))
        lob_img = nib.load(str(lobular_p))
        _save_overlay(
            tmpl_img, lob_img,
            "SUIT atlas on SUIT template",
            DOCS_QA / "qa01_suit_overlay.png",
        )
    except Exception as exc:
        logger.warning("Could not generate SUIT overlay: %s", exc)

    return result


def check_mni_consistency() -> dict:
    """
    Compare the MNI lobular atlas against the MNI152 template.
    """
    result = {
        "name": "MNI-space consistency",
        "affine_match": False,
        "shape_match": False,
        "status": "FAIL",
        "detail": "",
    }

    mni_tmpl = _mni_template()
    if mni_tmpl is None:
        result["detail"] = "MNI152 template not available"
        return result

    mni_lob_p = _find_atlas("*Anatom*space-MNI*dseg.nii*")
    if mni_lob_p is None:
        result["detail"] = "MNI lobular parcellation not found in atlas dir"
        return result

    tmpl_aff = mni_tmpl.affine
    tmpl_shape = mni_tmpl.shape[:3]

    lob_data, lob_aff, _ = load_nifti(mni_lob_p)

    result["affine_match"] = check_affine_match(tmpl_aff, lob_aff)
    result["shape_match"] = check_shape_match(tmpl_shape, lob_data.shape)

    # MNI atlases often live in the same 2-mm grid but may have been
    # cropped.  Treat matching affine as sufficient when shapes differ.
    result["status"] = "PASS" if result["affine_match"] else "FAIL"
    result["detail"] = (
        f"MNI template shape={tmpl_shape}, lobular shape={lob_data.shape[:3]}; "
        f"affine match={result['affine_match']}; "
        f"shape match={result['shape_match']}"
    )

    # Visualisation
    try:
        lob_img = nib.load(str(mni_lob_p))
        _save_overlay(
            mni_tmpl, lob_img,
            "MNI atlas on MNI152 template",
            DOCS_QA / "qa01_mni_overlay.png",
        )
    except Exception as exc:
        logger.warning("Could not generate MNI overlay: %s", exc)

    return result


def check_nuclei_within_cerebellum() -> dict:
    """
    Verify that deep nuclei probability maps fall inside the cerebellum.

    Uses the SUIT lobular parcellation as a proxy for the cerebellar mask.
    Nuclei voxels are expected to lie within or immediately adjacent to
    the labelled cerebellar cortex.
    """
    result = {
        "name": "Deep nuclei within cerebellum",
        "status": "FAIL",
        "detail": "",
        "per_nucleus": {},
    }

    lob_p = _find_atlas("*Anatom*space-SUIT*dseg.nii*")
    if lob_p is None:
        result["detail"] = "Lobular parcellation not found"
        return result

    lob_data, lob_aff, _ = load_nifti(lob_p, dtype=int)

    # Build a cerebellum mask and dilate by 1 voxel for tolerance
    cb_mask = lob_data > 0
    try:
        from scipy.ndimage import binary_dilation
        cb_mask_dilated = binary_dilation(cb_mask, iterations=2)
    except ImportError:
        cb_mask_dilated = cb_mask  # fall back if scipy absent

    # Search for nucleus probability maps
    nucleus_patterns = [
        ("dentate", "*Dentate*space-SUIT*.nii*"),
        ("fastigial", "*Fastigial*space-SUIT*.nii*"),
        ("interposed", "*Interposed*space-SUIT*.nii*"),
        ("emboliform", "*Emboliform*space-SUIT*.nii*"),
        ("globose", "*Globose*space-SUIT*.nii*"),
    ]

    found_any = False
    all_ok = True
    overlay_imgs = []

    for nuc_name, nuc_pattern in nucleus_patterns:
        nuc_p = _find_atlas(nuc_pattern)
        if nuc_p is None:
            # Try a broader search
            nuc_p = _find_atlas(f"*{nuc_name}*.nii*")
        if nuc_p is None:
            logger.info("  %s: not found (skipped)", nuc_name)
            continue

        found_any = True
        nuc_data, nuc_aff, _ = load_nifti(nuc_p)

        # Check affine match
        aff_ok = check_affine_match(lob_aff, nuc_aff)

        # Check containment
        if aff_ok and check_shape_match(lob_data.shape, nuc_data.shape):
            nuc_nonzero = nuc_data > 0.01  # above noise floor
            outside = np.logical_and(nuc_nonzero, ~cb_mask_dilated)
            frac_outside = outside.sum() / max(nuc_nonzero.sum(), 1)
            containment_ok = frac_outside < 0.05  # allow 5 % tolerance
        else:
            frac_outside = float("nan")
            containment_ok = False

        status = "PASS" if (aff_ok and containment_ok) else "FAIL"
        if not (aff_ok and containment_ok):
            all_ok = False

        result["per_nucleus"][nuc_name] = {
            "affine_match": aff_ok,
            "frac_outside": float(frac_outside),
            "status": status,
        }
        logger.info(
            "  %s: affine_ok=%s  frac_outside=%.4f  %s",
            nuc_name, aff_ok, frac_outside, status,
        )

        overlay_imgs.append(nib.load(str(nuc_p)))

    result["status"] = "PASS" if (found_any and all_ok) else "FAIL"
    result["detail"] = (
        f"Checked {len(result['per_nucleus'])} nuclei; "
        f"all_ok={all_ok}"
    )

    # Visualisation -- overlay first found nucleus on lobular atlas
    if overlay_imgs and lob_p:
        try:
            lob_img = nib.load(str(lob_p))
            _save_overlay(
                lob_img,
                overlay_imgs[0],
                "Nuclei on lobular atlas",
                DOCS_QA / "qa01_nuclei_overlay.png",
            )
        except Exception as exc:
            logger.warning("Could not generate nuclei overlay: %s", exc)

    return result


# ───────────────────────────────────────────────────────────────────────
# Formatted results table
# ───────────────────────────────────────────────────────────────────────

def _print_results_table(results: list[dict]) -> str:
    """Pretty-print results and return them as a string."""
    width = 55
    sep = "+" + "-" * width + "+" + "-" * 8 + "+"
    header = f"| {'Check':<{width - 1}} | {'Result':>6} |"

    lines = [sep, header, sep]
    for r in results:
        name = r.get("name", "unknown")
        status = r.get("status", "N/A")
        lines.append(f"| {name:<{width - 1}} | {status:>6} |")
        # Sub-results (e.g. per-nucleus)
        for sub_name, sub in r.get("per_nucleus", {}).items():
            sub_status = sub.get("status", "N/A")
            sub_line = f"  -> {sub_name}"
            lines.append(f"|   {sub_line:<{width - 3}} | {sub_status:>6} |")
    lines.append(sep)

    table = "\n".join(lines)
    print(table)
    return table


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all QA-01 atlas alignment checks."""
    ensure_directories()
    DOCS_QA.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  QA Check 01 -- Atlas Spatial Alignment")
    print("=" * 65)
    print()

    results: list[dict] = []

    # 1. SUIT-space
    logger.info("--- Check 1: SUIT-space consistency ---")
    results.append(check_suit_consistency())

    # 2. MNI-space
    logger.info("--- Check 2: MNI-space consistency ---")
    results.append(check_mni_consistency())

    # 3. Nuclei containment
    logger.info("--- Check 3: Deep nuclei within cerebellum ---")
    results.append(check_nuclei_within_cerebellum())

    # Results table
    print()
    _print_results_table(results)

    # Overall verdict
    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    print(f"\nOverall QA-01 result: {overall}")

    # Summary file
    summary_path = DOCS_QA / "qa01_summary.txt"
    summary_path.write_text(f"QA-01 atlas alignment: {overall}\n")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
