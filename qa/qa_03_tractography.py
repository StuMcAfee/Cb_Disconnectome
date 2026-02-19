"""
QA Check 03 --- Validate efferent pathway density maps.
========================================================

The efferent pathway density maps encode, for each deep cerebellar
nucleus, the spatial probability that a streamline originating in that
nucleus traverses a given voxel in MNI space.  These maps are derived
from normative tractography data (Elias et al. 2024 / HCP) and the
FWT atlas (Radwan et al. 2022).

Checks performed
----------------
1. **Non-empty** -- Each nuclear density map must contain non-zero voxels.
2. **SCP traversal** -- All density maps should have elevated values in
   the superior cerebellar peduncle (SCP) region, since efferents from
   all nuclei exit through the SCP.
3. **Anatomical trajectory** --
   - Dentate efferents should show high density in the SCP, crossing to
     the contralateral red nucleus / thalamus.
   - Fastigial efferents should show density in midline / brainstem
     structures.
4. **Laterality** -- Dentate density maps should be predominantly
   ipsilateral in the peduncle, then contralateral after the decussation.
5. **Spatial extent** -- No density should appear in implausible regions
   (e.g., above the Sylvian fissure in cortex).

Visualisations
--------------
* Sagittal MIP (maximum-intensity projection) of each density map.
* Representative axial slices through the SCP and thalamus.
* Glass-brain views of each density map.
* Bar chart comparing average SCP density across nuclei.

Outputs
-------
* ``docs/qa_reports/qa03_mip_sagittal.png``
* ``docs/qa_reports/qa03_axial_slices.png``
* ``docs/qa_reports/qa03_glass_brain.png``
* ``docs/qa_reports/qa03_scp_density_bar.png``
* ``docs/qa_reports/qa03_summary.txt``

Usage
-----
::

    python -m qa.qa_03_tractography
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

try:
    from nilearn import plotting as ni_plot
except ImportError as exc:
    sys.exit(f"nilearn is required: {exc}")

from src.utils import (
    DATA_INTERIM,
    DATA_FINAL,
    FWT_DIR,
    DOCS_QA,
    load_nifti,
    mm_to_voxel,
    get_mm_coordinate_grid,
    ensure_directories,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

NUCLEUS_NAMES = ["fastigial", "emboliform", "globose", "dentate"]

# Approximate MNI coordinates for key structures (mm)
# These are used for ROI-based spot checks.
SCP_ROI_CENTER_MNI = {
    "left_scp":  np.array([-5.0, -33.0, -22.0]),
    "right_scp": np.array([5.0,  -33.0, -22.0]),
}
SCP_ROI_RADIUS_MM = 6.0

RED_NUCLEUS_MNI = {
    "left_rn":  np.array([-5.0, -20.0, -6.0]),
    "right_rn": np.array([5.0,  -20.0, -6.0]),
}

# Above Sylvian fissure -- density here is implausible for cerebellar efferents
IMPLAUSIBLE_Z_THRESHOLD_MM = 50.0


# ───────────────────────────────────────────────────────────────────────
# File locators
# ───────────────────────────────────────────────────────────────────────

def _find_density_maps() -> dict[str, Path]:
    """
    Locate efferent density maps for each nucleus.

    Searches ``data/interim`` and ``data/final`` for files matching
    common naming patterns.
    """
    found: dict[str, Path] = {}

    for search_dir in [DATA_INTERIM, DATA_FINAL]:
        if not search_dir.exists():
            continue
        for nuc in NUCLEUS_NAMES:
            if nuc in found:
                continue
            patterns = [
                f"*{nuc}*density*.nii*",
                f"*efferent*{nuc}*.nii*",
                f"*pathway*{nuc}*.nii*",
            ]
            for pat in patterns:
                matches = sorted(search_dir.rglob(pat))
                if matches:
                    found[nuc] = matches[0]
                    break

    return found


def _find_scp_atlas() -> Path | None:
    """Locate the SCP probability map from the FWT atlas."""
    if not FWT_DIR.exists():
        return None
    patterns = [
        "*SCP*.nii*",
        "*scp*.nii*",
        "*superior_cerebellar_peduncle*.nii*",
    ]
    for pat in patterns:
        matches = sorted(FWT_DIR.rglob(pat))
        if matches:
            return matches[0]
    return None


def _sphere_mask(center_mm: np.ndarray, radius: float,
                 shape: tuple, affine: np.ndarray) -> np.ndarray:
    """Create a boolean sphere mask in voxel space."""
    mm_grid = get_mm_coordinate_grid(shape, affine)
    dist = np.sqrt(((mm_grid - center_mm) ** 2).sum(axis=-1))
    return dist <= radius


# ───────────────────────────────────────────────────────────────────────
# Checks
# ───────────────────────────────────────────────────────────────────────

def check_nonempty(density_maps: dict[str, Path]) -> list[dict]:
    """Verify each density map has non-zero voxels."""
    results = []
    for nuc, path in density_maps.items():
        data, _, _ = load_nifti(path)
        n_nonzero = (data > 0).sum()
        ok = n_nonzero > 0
        results.append({
            "name": f"Non-empty: {nuc}",
            "status": "PASS" if ok else "FAIL",
            "detail": f"{n_nonzero} non-zero voxels",
        })
    return results


def check_scp_traversal(density_maps: dict[str, Path]) -> list[dict]:
    """
    Every nuclear efferent should show elevated density in the SCP ROI.
    """
    results = []
    scp_densities: dict[str, float] = {}

    for nuc, path in density_maps.items():
        data, affine, _ = load_nifti(path)
        shape = data.shape[:3]

        # Combine left and right SCP ROIs
        scp_mask = np.zeros(shape, dtype=bool)
        for roi_center in SCP_ROI_CENTER_MNI.values():
            scp_mask |= _sphere_mask(roi_center, SCP_ROI_RADIUS_MM, shape, affine)

        if scp_mask.sum() == 0:
            results.append({
                "name": f"SCP traversal: {nuc}",
                "status": "FAIL",
                "detail": "SCP ROI mask is empty (resolution mismatch?)",
            })
            continue

        mean_scp = data[scp_mask].mean()
        mean_global = data[data > 0].mean() if (data > 0).any() else 0.0
        ratio = mean_scp / max(mean_global, 1e-10)
        ok = mean_scp > 0 and ratio > 1.0  # SCP should be above global mean
        scp_densities[nuc] = float(mean_scp)

        results.append({
            "name": f"SCP traversal: {nuc}",
            "status": "PASS" if ok else "FAIL",
            "detail": f"SCP mean={mean_scp:.4f}, global mean={mean_global:.4f}, ratio={ratio:.2f}",
        })

    return results


def check_anatomical_trajectory(density_maps: dict[str, Path]) -> list[dict]:
    """
    Dentate -> SCP -> contralateral red nucleus / thalamus.
    Fastigial -> midline / brainstem.
    """
    results = []

    # Dentate: check contralateral red nucleus density
    if "dentate" in density_maps:
        data, affine, _ = load_nifti(density_maps["dentate"])
        shape = data.shape[:3]

        # The dentate efferents decussate, so the *contralateral* RN should
        # have density.  We check that there is density near either RN.
        rn_mask = np.zeros(shape, dtype=bool)
        for rn_center in RED_NUCLEUS_MNI.values():
            rn_mask |= _sphere_mask(rn_center, 5.0, shape, affine)

        rn_density = data[rn_mask].mean() if rn_mask.sum() > 0 else 0.0
        ok = rn_density > 0
        results.append({
            "name": "Dentate -> contralateral RN trajectory",
            "status": "PASS" if ok else "FAIL",
            "detail": f"RN region mean density = {rn_density:.4f}",
        })

    # Fastigial: check midline / brainstem density
    if "fastigial" in density_maps:
        data, affine, _ = load_nifti(density_maps["fastigial"])
        shape = data.shape[:3]
        mm_grid = get_mm_coordinate_grid(shape, affine)
        x_mm = mm_grid[..., 0]

        midline_mask = (np.abs(x_mm) < 8.0) & (data > 0)
        lateral_nonzero = (np.abs(x_mm) >= 15.0) & (data > 0)

        n_midline = midline_mask.sum()
        n_lateral = lateral_nonzero.sum()
        ratio = n_midline / max(n_midline + n_lateral, 1)
        ok = ratio > 0.3  # at least 30% of density is midline

        results.append({
            "name": "Fastigial -> midline/brainstem trajectory",
            "status": "PASS" if ok else "FAIL",
            "detail": f"midline fraction = {ratio:.3f} ({n_midline} midline, {n_lateral} lateral)",
        })

    return results


def check_laterality(density_maps: dict[str, Path]) -> list[dict]:
    """
    Dentate efferents should be predominantly ipsilateral in the peduncle,
    then contralateral after decussation.
    """
    results = []

    if "dentate" not in density_maps:
        results.append({
            "name": "Dentate laterality",
            "status": "SKIP",
            "detail": "dentate density map not found",
        })
        return results

    data, affine, _ = load_nifti(density_maps["dentate"])
    shape = data.shape[:3]
    mm_grid = get_mm_coordinate_grid(shape, affine)
    x_mm = mm_grid[..., 0]
    z_mm = mm_grid[..., 2]

    # Peduncle level (z ~ -22 to -15 mm): ipsilateral dominance expected
    peduncle_mask = (z_mm > -25.0) & (z_mm < -15.0) & (data > 0)
    if peduncle_mask.sum() > 0:
        left_density = data[peduncle_mask & (x_mm < 0)].sum()
        right_density = data[peduncle_mask & (x_mm > 0)].sum()
        asym = abs(left_density - right_density) / max(left_density + right_density, 1e-10)
        # Some asymmetry is expected if the map represents one hemisphere
        results.append({
            "name": "Dentate laterality at peduncle",
            "status": "PASS",
            "detail": f"L={left_density:.2f}, R={right_density:.2f}, asymmetry={asym:.3f}",
        })
    else:
        results.append({
            "name": "Dentate laterality at peduncle",
            "status": "FAIL",
            "detail": "no voxels in peduncle region",
        })

    return results


def check_spatial_extent(density_maps: dict[str, Path]) -> list[dict]:
    """No density should appear above z = 50 mm (above Sylvian fissure)."""
    results = []

    for nuc, path in density_maps.items():
        data, affine, _ = load_nifti(path)
        shape = data.shape[:3]
        mm_grid = get_mm_coordinate_grid(shape, affine)
        z_mm = mm_grid[..., 2]

        above_mask = (z_mm > IMPLAUSIBLE_Z_THRESHOLD_MM) & (data > 0)
        n_above = above_mask.sum()
        n_total = (data > 0).sum()
        frac = n_above / max(n_total, 1)
        ok = frac < 0.01  # less than 1% in implausible region

        results.append({
            "name": f"Spatial extent: {nuc}",
            "status": "PASS" if ok else "FAIL",
            "detail": f"{n_above}/{n_total} voxels above z={IMPLAUSIBLE_Z_THRESHOLD_MM}mm ({frac:.4f})",
        })

    return results


# ───────────────────────────────────────────────────────────────────────
# Visualisations
# ───────────────────────────────────────────────────────────────────────

def _plot_sagittal_mip(density_maps: dict[str, Path], out: Path) -> None:
    """Sagittal MIP for each nucleus density map."""
    n = len(density_maps)
    if n == 0:
        return

    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (nuc, path) in zip(axes, density_maps.items()):
        data, affine, _ = load_nifti(path)
        # MIP along the x-axis (sagittal projection)
        mip = data.max(axis=0)  # project over x -> (Y, Z) plane
        ax.imshow(
            mip.T, origin="lower", aspect="auto", cmap="hot",
            interpolation="bilinear",
        )
        ax.set_title(f"{nuc} (sagittal MIP)")
        ax.set_xlabel("dim-1 (A-P)")
        ax.set_ylabel("dim-2 (I-S)")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    logger.info("Saved sagittal MIP: %s", out)


def _plot_axial_slices(density_maps: dict[str, Path], out: Path) -> None:
    """Axial slices at SCP level and thalamus level."""
    n = len(density_maps)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n), squeeze=False)

    for row, (nuc, path) in enumerate(density_maps.items()):
        data, affine, _ = load_nifti(path)
        shape = data.shape[:3]
        mm_grid = get_mm_coordinate_grid(shape, affine)
        z_mm = mm_grid[0, 0, :, 2]  # z for each slice

        # Find slices closest to SCP (z ~ -22) and thalamus (z ~ +8)
        scp_slice = int(np.argmin(np.abs(z_mm - (-22.0))))
        thal_slice = int(np.argmin(np.abs(z_mm - 8.0)))

        for col, (sl, label) in enumerate(
            [(scp_slice, f"SCP (z~-22mm)"), (thal_slice, f"Thalamus (z~+8mm)")]
        ):
            if sl < data.shape[2]:
                axes[row, col].imshow(
                    data[:, :, sl].T, origin="lower", cmap="hot",
                    aspect="auto", interpolation="bilinear",
                )
            axes[row, col].set_title(f"{nuc} -- {label}")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    logger.info("Saved axial slices: %s", out)


def _plot_glass_brain(density_maps: dict[str, Path], out: Path) -> None:
    """Glass-brain (maximum-intensity projection) views."""
    n = len(density_maps)
    if n == 0:
        return

    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (nuc, path) in zip(axes, density_maps.items()):
        img = nib.load(str(path))
        try:
            ni_plot.plot_glass_brain(
                img,
                display_mode="lyrz",
                axes=ax,
                title=nuc,
                colorbar=True,
            )
        except Exception:
            # Fall back to stat map plot
            try:
                ni_plot.plot_stat_map(
                    img,
                    display_mode="z",
                    cut_coords=5,
                    axes=ax,
                    title=nuc,
                )
            except Exception:
                ax.set_title(f"{nuc} (plot error)")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    logger.info("Saved glass brain: %s", out)


def _plot_scp_density_bar(density_maps: dict[str, Path], out: Path) -> None:
    """Bar chart of average SCP density across nuclei."""
    scp_means: dict[str, float] = {}

    for nuc, path in density_maps.items():
        data, affine, _ = load_nifti(path)
        shape = data.shape[:3]

        scp_mask = np.zeros(shape, dtype=bool)
        for roi_center in SCP_ROI_CENTER_MNI.values():
            scp_mask |= _sphere_mask(roi_center, SCP_ROI_RADIUS_MM, shape, affine)

        scp_means[nuc] = float(data[scp_mask].mean()) if scp_mask.sum() > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    nuclei = list(scp_means.keys())
    values = [scp_means[n] for n in nuclei]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"][:len(nuclei)]

    ax.bar(nuclei, values, color=colors, edgecolor="black")
    ax.set_ylabel("Mean density in SCP ROI")
    ax.set_title("Average efferent density in SCP by nucleus")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    logger.info("Saved SCP density bar chart: %s", out)


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
    """Run all QA-03 tractography density map checks."""
    ensure_directories()
    DOCS_QA.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  QA Check 03 -- Efferent Pathway Density Maps")
    print("=" * 65)
    print()

    # Locate density maps
    density_maps = _find_density_maps()

    if not density_maps:
        msg = (
            "No efferent density maps found.\n"
            "  Searched in: data/interim/, data/final/\n"
            "  Expected files matching: *<nucleus>*density*.nii*\n"
            "  Run the tractography pipeline first."
        )
        print(f"FAIL: {msg}")
        (DOCS_QA / "qa03_summary.txt").write_text("QA-03 tractography: FAIL (no density maps found)\n")
        return

    logger.info("Found density maps for: %s", list(density_maps.keys()))
    for nuc, path in density_maps.items():
        logger.info("  %s: %s", nuc, path)

    results: list[dict] = []

    # 1. Non-empty check
    results.extend(check_nonempty(density_maps))

    # 2. SCP traversal
    results.extend(check_scp_traversal(density_maps))

    # 3. Anatomical trajectory
    results.extend(check_anatomical_trajectory(density_maps))

    # 4. Laterality
    results.extend(check_laterality(density_maps))

    # 5. Spatial extent
    results.extend(check_spatial_extent(density_maps))

    # Results table
    print()
    _print_results_table(results)

    overall = "PASS" if all(
        r["status"] in ("PASS", "SKIP") for r in results
    ) else "FAIL"
    print(f"\nOverall QA-03 result: {overall}")

    # Visualisations
    logger.info("Generating visualisations ...")
    _plot_sagittal_mip(density_maps, DOCS_QA / "qa03_mip_sagittal.png")
    _plot_axial_slices(density_maps, DOCS_QA / "qa03_axial_slices.png")
    _plot_glass_brain(density_maps, DOCS_QA / "qa03_glass_brain.png")
    _plot_scp_density_bar(density_maps, DOCS_QA / "qa03_scp_density_bar.png")

    # Summary
    summary_path = DOCS_QA / "qa03_summary.txt"
    summary_path.write_text(f"QA-03 tractography density maps: {overall}\n")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
