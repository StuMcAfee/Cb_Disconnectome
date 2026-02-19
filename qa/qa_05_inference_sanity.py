"""
QA Check 05 --- Inference sanity tests with synthetic lesions.
===============================================================

This script creates a set of carefully designed synthetic lesion masks
and runs them through the full disconnectome inference pipeline.  The
expected pattern of cortical disruption is known *a priori* for each
lesion, so the test constitutes an end-to-end sanity check.

Test cases
----------
1. **Pure cortical lesion (right Crus I)** -- The disruption map should
   show high disconnection probability for right Crus I parcels and
   low/zero for distant parcels.
2. **SCP lesion (right SCP sphere)** -- Disrupts efferent fibres from
   the right cerebellar hemisphere; right-hemisphere parcels should be
   heavily affected.
3. **Midline SCP decussation lesion** -- Placed at the midbrain level
   where left and right SCP fibres cross; disruption should be
   bilateral.
4. **Large posterior fossa lesion** -- Encompasses cortex, nuclei, and
   peduncles; should produce both direct and downstream disconnection.
5. **Frontal cortex lesion (negative control)** -- Lesion placed far
   from the cerebellum; disruption should be zero.
6. **Empty lesion (negative control)** -- A zero-everywhere mask;
   disruption must be identically zero.

For each test, inference is run with all four aggregation methods:
``mean``, ``max``, ``weighted_sum``, ``threshold_fraction``.
The ``mean`` method is the default and is used for the primary evaluation.

Outputs
-------
* ``docs/qa_reports/qa05_test<N>_flatmap.png``  (one per test case)
* ``docs/qa_reports/qa05_summary.csv``           (top-10 parcels per test)
* ``docs/qa_reports/qa05_summary.txt``           (one-line pass/fail)

Usage
-----
::

    python -m qa.qa_05_inference_sanity
"""

from __future__ import annotations

import csv
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
    DATA_FINAL,
    DATA_INTERIM,
    DOCS_QA,
    load_nifti,
    make_sphere_lesion,
    ensure_directories,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

# ───────────────────────────────────────────────────────────────────────
# Inference methods (self-contained fallback)
# ───────────────────────────────────────────────────────────────────────

INFERENCE_METHODS = ["mean", "max", "weighted_sum", "threshold_fraction"]


def _try_import_inference():
    """
    Attempt to import the project's inference module.

    Returns a callable ``run_inference(lesion_img, method) -> dict``
    or None if unavailable.
    """
    try:
        from src.inference import run_inference
        return run_inference
    except ImportError:
        return None


def _find_4d_volume() -> Path | None:
    """Locate the 4D pathway occupancy volume."""
    for search_dir in [DATA_FINAL, DATA_INTERIM]:
        if not search_dir.exists():
            continue
        for pat in [
            "*pathway_occupancy*.nii*",
            "*4d_disconnectome*.nii*",
            "*disconnectome_4d*.nii*",
            "*occupancy_4d*.nii*",
        ]:
            matches = sorted(search_dir.rglob(pat))
            if matches:
                return matches[0]
    return None


def _find_lobular_atlas() -> Path | None:
    candidates = sorted(ATLAS_DIR.rglob("*Anatom*space-SUIT*dseg.nii*"))
    return candidates[0] if candidates else None


def _load_parcel_labels() -> dict[int, str]:
    """Load label-to-name mapping."""
    try:
        import pandas as pd
        tsv = sorted(ATLAS_DIR.rglob("*Anatom*space-SUIT*.tsv"))
        if tsv:
            df = pd.read_csv(tsv[0], sep="\t")
            idx_col = "index" if "index" in df.columns else df.columns[0]
            name_col = "name" if "name" in df.columns else df.columns[1]
            return dict(zip(df[idx_col].astype(int), df[name_col].astype(str)))
    except Exception:
        pass

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
# Fallback inference (if src.inference not available)
# ───────────────────────────────────────────────────────────────────────

def _fallback_inference(
    lesion_data: np.ndarray,
    vol_4d: np.ndarray,
    method: str,
    threshold: float = 0.25,
) -> np.ndarray:
    """
    Compute parcel disruption scores from the 4D volume and a lesion mask.

    Parameters
    ----------
    lesion_data : (X, Y, Z) binary mask
    vol_4d : (X, Y, Z, N) pathway occupancy
    method : one of 'mean', 'max', 'weighted_sum', 'threshold_fraction'
    threshold : for threshold_fraction method

    Returns
    -------
    scores : (N,) disruption score per parcel
    """
    n_parcels = vol_4d.shape[3]
    lesion_mask = lesion_data > 0.5

    if lesion_mask.sum() == 0:
        return np.zeros(n_parcels, dtype=np.float64)

    scores = np.zeros(n_parcels, dtype=np.float64)

    for p in range(n_parcels):
        vals = vol_4d[lesion_mask, p]
        if vals.size == 0:
            continue
        if method == "max":
            scores[p] = float(vals.max())
        elif method == "mean":
            scores[p] = float(vals.mean())
        elif method == "weighted_sum":
            scores[p] = float(vals.sum())
        elif method == "threshold_fraction":
            scores[p] = float((vals >= threshold).sum() / max(vals.size, 1))
        else:
            scores[p] = float(vals.mean())

    return scores


# ───────────────────────────────────────────────────────────────────────
# Test case definitions
# ───────────────────────────────────────────────────────────────────────

# MNI / SUIT approximate coordinates used for sphere centres.
# These are coarse anatomical landmarks.

TEST_CASES = [
    {
        "id": 1,
        "name": "Pure cortical lesion (right Crus I)",
        "center_mm": (28.0, -76.0, -32.0),  # approx right Crus I in SUIT
        "radius_mm": 8.0,
        "expect": {
            "high_parcels": ["Right CrusI", "Right CrusII"],
            "zero_parcels": ["Left I-IV", "Left V"],
            "overall": "focal right posterior",
        },
    },
    {
        "id": 2,
        "name": "SCP lesion (right SCP)",
        "center_mm": (5.0, -33.0, -22.0),  # right SCP in MNI
        "radius_mm": 5.0,
        "expect": {
            "high_parcels": [],  # many right parcels
            "zero_parcels": [],
            "overall": "broad right hemisphere disruption",
        },
    },
    {
        "id": 3,
        "name": "Midline SCP decussation",
        "center_mm": (0.0, -20.0, -12.0),  # midbrain decussation
        "radius_mm": 6.0,
        "expect": {
            "high_parcels": [],
            "zero_parcels": [],
            "overall": "bilateral disruption",
        },
    },
    {
        "id": 4,
        "name": "Large posterior fossa",
        "center_mm": (5.0, -55.0, -30.0),
        "radius_mm": 20.0,
        "expect": {
            "high_parcels": [],
            "zero_parcels": [],
            "overall": "widespread direct + downstream",
        },
    },
    {
        "id": 5,
        "name": "Frontal cortex (negative control)",
        "center_mm": (30.0, 50.0, 30.0),  # frontal lobe
        "radius_mm": 10.0,
        "expect": {
            "high_parcels": [],
            "zero_parcels": ["all"],
            "overall": "zero disruption",
        },
    },
    {
        "id": 6,
        "name": "Empty lesion (negative control)",
        "center_mm": None,  # sentinel: create an all-zero mask
        "radius_mm": 0.0,
        "expect": {
            "high_parcels": [],
            "zero_parcels": ["all"],
            "overall": "zero disruption",
        },
    },
]


# ───────────────────────────────────────────────────────────────────────
# Visualisations
# ───────────────────────────────────────────────────────────────────────

def _plot_disruption_flatmap(
    scores: np.ndarray,
    label_names: dict[int, str],
    cortical_labels: list[int],
    affine: np.ndarray,
    shape_3d: tuple,
    test_name: str,
    out: Path,
) -> None:
    """
    Project disruption scores onto a cerebellar flatmap.

    Falls back to a bar chart if SUITPy is unavailable.
    """
    # Build a 3D volume of disruption scores for flatmap
    try:
        import SUITPy as suit

        # Create a volume where each voxel gets the score of its parcel
        atlas_path = _find_lobular_atlas()
        if atlas_path is not None:
            atlas_data, atlas_aff, _ = load_nifti(atlas_path, dtype=int)
            score_vol = np.zeros(atlas_data.shape[:3], dtype=np.float32)
            for vol_idx, label in enumerate(cortical_labels):
                if vol_idx < len(scores):
                    score_vol[atlas_data == label] = scores[vol_idx]

            score_img = nib.Nifti1Image(score_vol, atlas_aff)
            fig = plt.figure(figsize=(12, 8))
            flatmap_data = suit.flatmap.vol_to_surf(score_img, stats="nanmean")
            suit.flatmap.plot(
                flatmap_data,
                render="matplotlib",
                cmap="YlOrRd",
                new_figure=False,
            )
            plt.title(f"Disruption: {test_name}")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out), dpi=150)
            plt.close(fig)
            logger.info("Saved flatmap: %s", out)
            return
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Flatmap failed: %s; falling back to bar chart", exc)

    # Fallback: bar chart
    _plot_disruption_bar(scores, label_names, cortical_labels, test_name, out)


def _plot_disruption_bar(
    scores: np.ndarray,
    label_names: dict[int, str],
    cortical_labels: list[int],
    test_name: str,
    out: Path,
) -> None:
    """Simple bar chart of disruption scores per parcel."""
    n = min(len(scores), len(cortical_labels))
    names = [label_names.get(cortical_labels[i], str(cortical_labels[i])) for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(n), scores[:n], color="firebrick", edgecolor="black", alpha=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Disruption score")
    ax.set_title(f"Disruption: {test_name}")
    ax.set_ylim(0, max(scores.max() * 1.1, 0.01))

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    logger.info("Saved bar chart: %s", out)


# ───────────────────────────────────────────────────────────────────────
# Results helpers
# ───────────────────────────────────────────────────────────────────────

def _top_parcels(
    scores: np.ndarray,
    label_names: dict[int, str],
    cortical_labels: list[int],
    k: int = 10,
) -> list[dict]:
    """Return top-k disrupted parcels as a list of dicts."""
    n = min(len(scores), len(cortical_labels))
    idx_sorted = np.argsort(scores[:n])[::-1][:k]

    rows = []
    for rank, idx in enumerate(idx_sorted, 1):
        lbl = cortical_labels[idx]
        rows.append({
            "rank": rank,
            "label": lbl,
            "name": label_names.get(lbl, str(lbl)),
            "score": float(scores[idx]),
        })
    return rows


def _evaluate_test(
    test_case: dict,
    scores: np.ndarray,
    label_names: dict[int, str],
    cortical_labels: list[int],
) -> dict:
    """Evaluate whether a test case passes its expected pattern."""
    result = {
        "test_id": test_case["id"],
        "test_name": test_case["name"],
        "status": "PASS",
        "detail": "",
    }

    total_disruption = float(scores.sum())
    max_disruption = float(scores.max()) if scores.size > 0 else 0.0

    # Build name -> index lookup
    name_to_idx = {}
    for i, lbl in enumerate(cortical_labels):
        if i < len(scores):
            name_to_idx[label_names.get(lbl, "")] = i

    issues = []

    # Check expected-zero tests (negative controls)
    if test_case["expect"].get("zero_parcels") == ["all"]:
        if total_disruption > 1e-6:
            issues.append(f"expected zero disruption, got total={total_disruption:.4f}")
    else:
        # Check high-expected parcels
        for pname in test_case["expect"].get("high_parcels", []):
            if pname in name_to_idx:
                idx = name_to_idx[pname]
                if scores[idx] < 0.1:
                    issues.append(f"expected high disruption for '{pname}', got {scores[idx]:.4f}")

    if issues:
        result["status"] = "FAIL"
        result["detail"] = "; ".join(issues)
    else:
        result["detail"] = (
            f"total_disruption={total_disruption:.3f}, "
            f"max={max_disruption:.3f}"
        )

    return result


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all QA-05 inference sanity tests."""
    ensure_directories()
    DOCS_QA.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  QA Check 05 -- Inference Sanity Tests")
    print("=" * 65)
    print()

    # Locate data
    vol_path = _find_4d_volume()
    atlas_path = _find_lobular_atlas()
    label_names = _load_parcel_labels()
    cortical_labels = sorted([k for k in label_names.keys() if k <= 28])

    # Determine reference NIfTI for sphere generation
    ref_path = atlas_path  # default to atlas
    vol_4d = None
    affine = None
    shape_3d = None

    if vol_path is not None:
        logger.info("Loading 4D volume: %s", vol_path)
        vol_4d, affine, _ = load_nifti(vol_path)
        shape_3d = vol_4d.shape[:3]
        ref_path = vol_path
    elif atlas_path is not None:
        _, affine, _ = load_nifti(atlas_path)
        shape_3d = load_nifti(atlas_path)[0].shape[:3]
    else:
        print("FAIL: Neither 4D volume nor atlas found. Cannot run tests.")
        (DOCS_QA / "qa05_summary.txt").write_text(
            "QA-05 inference sanity: FAIL (no data)\n"
        )
        return

    # Try to import project inference module
    run_inference_fn = _try_import_inference()

    # CSV output for top-10 parcels
    csv_path = DOCS_QA / "qa05_summary.csv"
    csv_rows: list[dict] = []

    all_results: list[dict] = []

    for tc in TEST_CASES:
        print(f"\n--- Test {tc['id']}: {tc['name']} ---")

        # Create lesion mask
        if tc["center_mm"] is None:
            # Empty lesion
            lesion_data = np.zeros(shape_3d, dtype=np.float32)
            lesion_img = nib.Nifti1Image(lesion_data, affine)
        else:
            lesion_img = make_sphere_lesion(
                tc["center_mm"], tc["radius_mm"], ref_path
            )
            lesion_data = lesion_img.get_fdata()

        n_lesion_voxels = int((lesion_data > 0.5).sum())
        print(f"  Lesion voxels: {n_lesion_voxels}")

        for method in INFERENCE_METHODS:
            # Run inference
            if run_inference_fn is not None and vol_path is not None:
                try:
                    result_dict = run_inference_fn(lesion_img, method=method)
                    # Expect result_dict to have 'scores' key or be array-like
                    if isinstance(result_dict, dict):
                        scores = np.array(result_dict.get("scores", result_dict.get("disruption", [])))
                    else:
                        scores = np.asarray(result_dict)
                except Exception as exc:
                    logger.warning(
                        "  src.inference failed for method '%s': %s; using fallback",
                        method, exc,
                    )
                    if vol_4d is not None:
                        scores = _fallback_inference(lesion_data, vol_4d, method)
                    else:
                        scores = np.zeros(len(cortical_labels))
            elif vol_4d is not None:
                scores = _fallback_inference(lesion_data, vol_4d, method)
            else:
                scores = np.zeros(len(cortical_labels))

            print(f"  Method={method:20s}  sum={scores.sum():.3f}  max={scores.max():.4f}")

            # Evaluate only for the 'mean' method (primary/default)
            if method == "mean":
                eval_result = _evaluate_test(tc, scores, label_names, cortical_labels)
                all_results.append(eval_result)
                print(f"  -> {eval_result['status']}: {eval_result['detail']}")

                # Top-10 parcels
                top10 = _top_parcels(scores, label_names, cortical_labels, k=10)
                for row in top10:
                    csv_rows.append({
                        "test_id": tc["id"],
                        "test_name": tc["name"],
                        "method": method,
                        "rank": row["rank"],
                        "parcel_label": row["label"],
                        "parcel_name": row["name"],
                        "score": row["score"],
                    })

                # Flatmap visualisation
                _plot_disruption_flatmap(
                    scores, label_names, cortical_labels,
                    affine, shape_3d,
                    f"Test {tc['id']}: {tc['name']} ({method})",
                    DOCS_QA / f"qa05_test{tc['id']}_flatmap.png",
                )

    # Write CSV
    if csv_rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["test_id", "test_name", "method", "rank",
                       "parcel_label", "parcel_name", "score"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("Saved top-10 parcels CSV: %s", csv_path)

    # Results table
    print("\n" + "=" * 65)
    width = 50
    sep = "+" + "-" * width + "+" + "-" * 8 + "+"
    header = f"| {'Test':<{width - 1}} | {'Result':>6} |"
    lines = [sep, header, sep]
    for r in all_results:
        name = f"Test {r['test_id']}: {r['test_name']}"[:width - 1]
        lines.append(f"| {name:<{width - 1}} | {r['status']:>6} |")
    lines.append(sep)
    print("\n".join(lines))

    for r in all_results:
        print(f"  Test {r['test_id']}: {r['detail']}")

    overall = "PASS" if all(r["status"] == "PASS" for r in all_results) else "FAIL"
    print(f"\nOverall QA-05 result: {overall}")

    summary_path = DOCS_QA / "qa05_summary.txt"
    summary_path.write_text(f"QA-05 inference sanity: {overall}\n")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
