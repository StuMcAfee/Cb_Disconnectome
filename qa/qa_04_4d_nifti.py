"""
QA Check 04 --- Validate the assembled 4D pathway occupancy volume.
====================================================================

The 4D pathway occupancy volume is the central data structure of the
disconnectome model.  It has shape ``(X, Y, Z, N_parcels)`` where each
3D sub-volume encodes the probability that a streamline connecting to
cortical parcel *p* passes through voxel ``(i, j, k)``.

Checks performed
----------------
1. **Shape** -- The 4th dimension must match the expected parcel count
   from the SUIT parcellation.
2. **Value range** -- All values must lie in [0, 1].
3. **Direct injury encoding** -- For each parcel, voxels that fall
   *inside* that cortical region should have value 1.0 (representing
   certain disconnection by a lesion at that location).
4. **Self-consistency** -- Adjacent parcels (e.g., Left CrusI and Left
   CrusII) should have more correlated pathway maps than distant parcels
   (e.g., Left I-IV vs Right IX).
5. **Asymmetry check** -- Left and right hemisphere mirrors should show
   reasonable symmetry (correlation > 0.3).
6. **Sparsity** -- Flag any parcel volume with more than 30 % non-zero
   voxels, which may indicate a processing error.

Visualisations
--------------
* Parcel-to-parcel correlation matrix (heatmap).
* Representative axial montage for selected parcels.
* Total-density flatmap (sum across parcels).

Outputs
-------
* ``docs/qa_reports/qa04_correlation_matrix.png``
* ``docs/qa_reports/qa04_axial_montage.png``
* ``docs/qa_reports/qa04_density_flatmap.png``
* ``docs/qa_reports/qa04_summary.txt``

Usage
-----
::

    python -m qa.qa_04_4d_nifti
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
    import seaborn as sns
except ImportError:
    sns = None  # degrade gracefully

from src.utils import (
    ATLAS_DIR,
    DATA_INTERIM,
    DATA_FINAL,
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
# File locators
# ───────────────────────────────────────────────────────────────────────

def _find_4d_volume() -> Path | None:
    """Locate the 4D pathway occupancy NIfTI."""
    for search_dir in [DATA_FINAL, DATA_INTERIM]:
        if not search_dir.exists():
            continue
        patterns = [
            "*pathway_occupancy*.nii*",
            "*4d_disconnectome*.nii*",
            "*disconnectome_4d*.nii*",
            "*occupancy_4d*.nii*",
        ]
        for pat in patterns:
            matches = sorted(search_dir.rglob(pat))
            if matches:
                return matches[0]
    return None


def _find_lobular_atlas() -> Path | None:
    candidates = sorted(ATLAS_DIR.rglob("*Anatom*space-SUIT*dseg.nii*"))
    return candidates[0] if candidates else None


def _load_parcel_labels() -> dict[int, str]:
    """Load integer-to-name label mapping (fallback to defaults)."""
    try:
        import pandas as pd
        tsv_candidates = sorted(ATLAS_DIR.rglob("*Anatom*space-SUIT*.tsv"))
        if tsv_candidates:
            df = pd.read_csv(tsv_candidates[0], sep="\t")
            idx_col = "index" if "index" in df.columns else df.columns[0]
            name_col = "name" if "name" in df.columns else df.columns[1]
            return dict(zip(df[idx_col].astype(int), df[name_col].astype(str)))
    except Exception:
        pass

    # Default SUIT labels (cortical lobules 1-28)
    return {
        1: "Left I-IV", 2: "Right I-IV", 3: "Left V", 4: "Right V",
        5: "Left VI", 6: "Vermis VI", 7: "Right VI",
        8: "Left CrusI", 9: "Vermis CrusI", 10: "Right CrusI",
        11: "Left CrusII", 12: "Vermis CrusII", 13: "Right CrusII",
        14: "Left VIIb", 15: "Vermis VIIb", 16: "Right VIIb",
        17: "Left VIIIa", 18: "Vermis VIIIa", 19: "Right VIIIa",
        20: "Left VIIIb", 21: "Vermis VIIIb", 22: "Right VIIIb",
        23: "Left IX", 24: "Vermis IX", 25: "Right IX",
        26: "Left X", 27: "Vermis X", 28: "Right X",
    }


# ───────────────────────────────────────────────────────────────────────
# Checks
# ───────────────────────────────────────────────────────────────────────

def check_shape(data: np.ndarray, expected_parcels: int | None) -> dict:
    """4th dimension matches expected parcel count."""
    if data.ndim != 4:
        return {
            "name": "Shape (4D volume)",
            "status": "FAIL",
            "detail": f"expected 4D, got {data.ndim}D with shape {data.shape}",
        }
    n_parcels = data.shape[3]
    if expected_parcels is not None and n_parcels != expected_parcels:
        return {
            "name": "Shape (parcel count)",
            "status": "FAIL",
            "detail": f"4th dim = {n_parcels}, expected {expected_parcels}",
        }
    return {
        "name": "Shape (4D volume)",
        "status": "PASS",
        "detail": f"shape = {data.shape}, parcels = {n_parcels}",
    }


def check_value_range(data: np.ndarray) -> dict:
    """All values should lie in [0, 1]."""
    vmin, vmax = float(data.min()), float(data.max())
    ok = vmin >= -1e-7 and vmax <= 1.0 + 1e-7
    return {
        "name": "Value range [0, 1]",
        "status": "PASS" if ok else "FAIL",
        "detail": f"min={vmin:.6f}, max={vmax:.6f}",
    }


def check_direct_injury(
    data_4d: np.ndarray,
    atlas_data: np.ndarray,
    cortical_labels: list[int],
) -> dict:
    """
    Cortical voxels should have value 1.0 in their corresponding parcel
    volume.
    """
    n_checked = 0
    n_correct = 0
    issues = []

    for vol_idx, label in enumerate(cortical_labels):
        if vol_idx >= data_4d.shape[3]:
            break
        mask = atlas_data == label
        if mask.sum() == 0:
            continue

        vals_in_parcel = data_4d[mask, vol_idx]
        n_checked += vals_in_parcel.size
        # Direct injury -> value should be 1.0
        correct = np.isclose(vals_in_parcel, 1.0, atol=1e-3)
        n_correct += correct.sum()

        frac_correct = correct.sum() / vals_in_parcel.size
        if frac_correct < 0.95:
            issues.append(
                f"label {label} vol {vol_idx}: {frac_correct:.1%} at 1.0"
            )

    fraction = n_correct / max(n_checked, 1)
    ok = fraction >= 0.95

    return {
        "name": "Direct injury encoding (cortex = 1.0)",
        "status": "PASS" if ok else "FAIL",
        "detail": (
            f"{n_correct}/{n_checked} voxels correct ({fraction:.4f})"
            + (f"; issues: {issues[:3]}" if issues else "")
        ),
    }


def check_self_consistency(data_4d: np.ndarray) -> dict:
    """
    Adjacent parcels should have more correlated maps than distant ones.

    We flatten each parcel volume and compute the correlation matrix.
    Adjacent parcels (consecutive indices) should have higher average
    correlation than the global average.
    """
    n = data_4d.shape[3]
    if n < 3:
        return {
            "name": "Self-consistency (adjacent correlation)",
            "status": "SKIP",
            "detail": "fewer than 3 parcels",
        }

    # Flatten each volume to a 1D vector and compute correlation
    flat = data_4d.reshape(-1, n)  # (V, N)
    # Remove zero-variance columns
    stds = flat.std(axis=0)
    valid = stds > 1e-10
    if valid.sum() < 3:
        return {
            "name": "Self-consistency (adjacent correlation)",
            "status": "SKIP",
            "detail": "too few non-constant parcel volumes",
        }

    corr = np.corrcoef(flat[:, valid].T)

    # Adjacent correlation (off-diagonal +/- 1)
    diag_1 = np.array([corr[i, i + 1] for i in range(corr.shape[0] - 1)])
    mean_adj = float(np.nanmean(diag_1))

    # Global mean (upper triangle, excluding diagonal and adjacent)
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=2)
    mean_global = float(np.nanmean(corr[mask_upper])) if mask_upper.sum() > 0 else 0.0

    ok = mean_adj > mean_global

    return {
        "name": "Self-consistency (adjacent > global correlation)",
        "status": "PASS" if ok else "FAIL",
        "detail": f"adj_mean_r={mean_adj:.3f}, global_mean_r={mean_global:.3f}",
    }


def check_asymmetry(data_4d: np.ndarray, label_names: dict[int, str],
                     cortical_labels: list[int]) -> dict:
    """
    Left/right mirror parcels should show correlation > 0.3.

    We attempt to pair parcels by name (e.g., 'Left V' <-> 'Right V').
    """
    # Build name -> volume index mapping
    name_to_idx: dict[str, int] = {}
    for vol_idx, label in enumerate(cortical_labels):
        if vol_idx >= data_4d.shape[3]:
            break
        name = label_names.get(label, f"label_{label}")
        name_to_idx[name] = vol_idx

    pairs_checked = 0
    pairs_ok = 0
    pair_corrs: list[float] = []

    for name, idx in name_to_idx.items():
        # Find mirror
        mirror_name = None
        if name.startswith("Left "):
            mirror_name = "Right " + name[5:]
        elif name.startswith("Right "):
            mirror_name = "Left " + name[6:]

        if mirror_name is None or mirror_name not in name_to_idx:
            continue

        mirror_idx = name_to_idx[mirror_name]
        if mirror_idx <= idx:  # avoid double-counting
            continue

        vol_a = data_4d[..., idx].ravel()
        vol_b = data_4d[..., mirror_idx].ravel()

        # Flip vol_b along x-axis for mirror comparison
        vol_b_3d = data_4d[..., mirror_idx]
        vol_b_flipped = vol_b_3d[::-1, :, :].ravel()

        # Correlation
        if vol_a.std() > 1e-10 and vol_b_flipped.std() > 1e-10:
            r = float(np.corrcoef(vol_a, vol_b_flipped)[0, 1])
        else:
            r = 0.0

        pairs_checked += 1
        pair_corrs.append(r)
        if r > 0.3:
            pairs_ok += 1

    if pairs_checked == 0:
        return {
            "name": "L/R asymmetry check",
            "status": "SKIP",
            "detail": "no L/R pairs found",
        }

    mean_r = float(np.mean(pair_corrs))
    ok = pairs_ok >= pairs_checked * 0.7  # 70% of pairs pass

    return {
        "name": "L/R mirror symmetry (r > 0.3)",
        "status": "PASS" if ok else "FAIL",
        "detail": f"{pairs_ok}/{pairs_checked} pairs pass, mean r={mean_r:.3f}",
    }


def check_sparsity(data_4d: np.ndarray) -> dict:
    """Flag volumes with > 30% non-zero voxels."""
    n = data_4d.shape[3]
    n_voxels = np.prod(data_4d.shape[:3])
    flagged = []

    for i in range(n):
        frac = (data_4d[..., i] > 0).sum() / n_voxels
        if frac > 0.30:
            flagged.append((i, frac))

    ok = len(flagged) == 0
    detail = f"{len(flagged)}/{n} volumes exceed 30% non-zero"
    if flagged:
        detail += f"; worst: vol {flagged[0][0]} ({flagged[0][1]:.1%})"

    return {
        "name": "Sparsity (no volume > 30% non-zero)",
        "status": "PASS" if ok else "WARN",
        "detail": detail,
    }


# ───────────────────────────────────────────────────────────────────────
# Visualisations
# ───────────────────────────────────────────────────────────────────────

def _plot_correlation_matrix(data_4d: np.ndarray, label_names: dict[int, str],
                              cortical_labels: list[int], out: Path) -> None:
    """Parcel-to-parcel correlation heatmap."""
    n = data_4d.shape[3]
    flat = data_4d.reshape(-1, n)
    stds = flat.std(axis=0)
    valid = stds > 1e-10

    if valid.sum() < 2:
        logger.warning("Fewer than 2 non-constant volumes; skipping correlation matrix.")
        return

    corr = np.corrcoef(flat[:, valid].T)

    # Labels for the valid volumes
    valid_indices = np.where(valid)[0]
    tick_labels = []
    for vi in valid_indices:
        if vi < len(cortical_labels):
            lbl = cortical_labels[vi]
            tick_labels.append(label_names.get(lbl, str(lbl)))
        else:
            tick_labels.append(str(vi))

    fig, ax = plt.subplots(figsize=(max(12, n * 0.5), max(10, n * 0.4)))

    if sns is not None:
        sns.heatmap(
            corr, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
            xticklabels=tick_labels, yticklabels=tick_labels,
            square=True, linewidths=0.3,
        )
    else:
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tick_labels)))
        ax.set_yticklabels(tick_labels, fontsize=6)
        plt.colorbar(im, ax=ax)

    ax.set_title("Parcel pathway-map correlation matrix")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    logger.info("Saved correlation matrix: %s", out)


def _plot_axial_montage(data_4d: np.ndarray, affine: np.ndarray,
                         label_names: dict[int, str],
                         cortical_labels: list[int], out: Path) -> None:
    """Axial montage for a selection of representative parcels."""
    n = data_4d.shape[3]
    # Pick up to 6 evenly spaced parcels
    indices = np.linspace(0, n - 1, min(n, 6), dtype=int)

    n_show = len(indices)
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3 * n_show), squeeze=False)

    for row, vi in enumerate(indices):
        vol = data_4d[..., vi]
        mid_z = vol.shape[2] // 2
        slices = [max(mid_z - 5, 0), mid_z, min(mid_z + 5, vol.shape[2] - 1)]

        lbl = cortical_labels[vi] if vi < len(cortical_labels) else vi
        name = label_names.get(lbl, str(lbl))

        for col, sz in enumerate(slices):
            axes[row, col].imshow(
                vol[:, :, sz].T, origin="lower", cmap="hot",
                aspect="auto", interpolation="bilinear", vmin=0, vmax=1,
            )
            axes[row, col].set_title(f"{name} z={sz}", fontsize=8)
            axes[row, col].axis("off")

    fig.suptitle("Pathway occupancy -- representative parcel axial slices", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    logger.info("Saved axial montage: %s", out)


def _plot_density_flatmap(data_4d: np.ndarray, affine: np.ndarray, out: Path) -> None:
    """Flatmap of the total (sum) density across all parcels."""
    try:
        import SUITPy as suit

        total = data_4d.sum(axis=-1)
        total_img = nib.Nifti1Image(total.astype(np.float32), affine)

        fig = plt.figure(figsize=(12, 8))
        flatmap_data = suit.flatmap.vol_to_surf(total_img, stats="nanmean")
        suit.flatmap.plot(
            flatmap_data,
            render="matplotlib",
            cmap="hot",
            new_figure=False,
        )
        plt.title("Total pathway occupancy -- flatmap")

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        logger.info("Saved density flatmap: %s", out)

    except ImportError:
        logger.warning("SUITPy not available -- skipping flatmap")
    except Exception as exc:
        logger.warning("Flatmap generation failed: %s", exc)


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
    """Run all QA-04 checks on the 4D pathway occupancy volume."""
    ensure_directories()
    DOCS_QA.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  QA Check 04 -- 4D Pathway Occupancy Volume")
    print("=" * 65)
    print()

    vol_path = _find_4d_volume()
    if vol_path is None:
        msg = (
            "4D pathway occupancy volume not found.\n"
            "  Searched: data/final/, data/interim/\n"
            "  Expected: *pathway_occupancy*.nii* or *disconnectome_4d*.nii*\n"
            "  Run the assembly pipeline first."
        )
        print(f"FAIL: {msg}")
        (DOCS_QA / "qa04_summary.txt").write_text("QA-04 4D volume: FAIL (file not found)\n")
        return

    logger.info("Loading 4D volume: %s", vol_path)
    data_4d, affine, _ = load_nifti(vol_path)

    # Load atlas for cross-referencing
    lob_path = _find_lobular_atlas()
    atlas_data = None
    label_names = _load_parcel_labels()
    cortical_labels = sorted([k for k in label_names.keys() if k <= 28])

    if lob_path is not None:
        atlas_data, _, _ = load_nifti(lob_path, dtype=int)

    expected_parcels = len(cortical_labels) if cortical_labels else None

    results: list[dict] = []

    # 1. Shape
    results.append(check_shape(data_4d, expected_parcels))

    # 2. Value range
    results.append(check_value_range(data_4d))

    # 3. Direct injury encoding
    if atlas_data is not None:
        results.append(check_direct_injury(data_4d, atlas_data, cortical_labels))
    else:
        results.append({
            "name": "Direct injury encoding",
            "status": "SKIP",
            "detail": "atlas not found for cross-reference",
        })

    # 4. Self-consistency
    results.append(check_self_consistency(data_4d))

    # 5. Asymmetry check
    results.append(check_asymmetry(data_4d, label_names, cortical_labels))

    # 6. Sparsity
    results.append(check_sparsity(data_4d))

    # Results table
    print()
    _print_results_table(results)

    overall = "PASS" if all(
        r["status"] in ("PASS", "SKIP", "WARN") for r in results
    ) else "FAIL"
    print(f"\nOverall QA-04 result: {overall}")

    # Visualisations
    logger.info("Generating visualisations ...")
    _plot_correlation_matrix(data_4d, label_names, cortical_labels,
                              DOCS_QA / "qa04_correlation_matrix.png")
    _plot_axial_montage(data_4d, affine, label_names, cortical_labels,
                         DOCS_QA / "qa04_axial_montage.png")
    _plot_density_flatmap(data_4d, affine, DOCS_QA / "qa04_density_flatmap.png")

    # Summary
    summary_path = DOCS_QA / "qa04_summary.txt"
    summary_path.write_text(f"QA-04 4D pathway occupancy: {overall}\n")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
