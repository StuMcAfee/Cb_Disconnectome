"""
Lesion inference engine for the cerebellar disconnectome model.

Takes a binary lesion mask in SUIT space and produces cortical disruption
probabilities by querying a precomputed 4D pathway occupancy volume.

Input:  Binary lesion mask in SUIT space (NIfTI, 3D)
Output: 1D array of disruption probabilities, one per cortical parcel

Aggregation methods:
  - max:                maximum pathway occupancy across lesion voxels
  - mean:              average pathway occupancy across lesion voxels
  - weighted_sum:      sum of occupancy values, normalized to [0, 1]
  - threshold_fraction: fraction of pathway volume intersected (threshold > 0.1)

Usage:
    python -m src.inference <lesion_SUIT.nii.gz> [output_dir]
"""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.utils import (
    load_nifti,
    save_nifti,
    load_metadata,
    check_affine_match,
    check_shape_match,
    DATA_FINAL,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Default file paths within data/final/
# ---------------------------------------------------------------------------

DEFAULT_OCCUPANCY_PATH = DATA_FINAL / "pathway_occupancy_4d.nii.gz"
DEFAULT_PARCELLATION_PATH = DATA_FINAL / "cortical_parcellation_SUIT.nii.gz"
DEFAULT_REFERENCE_PATH = DATA_FINAL / "SUIT_reference.nii.gz"
DEFAULT_METADATA_PATH = DATA_FINAL / "parcel_metadata.json"

# Occupancy threshold for the 'threshold_fraction' method
_THRESHOLD_FRACTION_CUTOFF = 0.1

# Available aggregation methods
VALID_METHODS = ("max", "mean", "weighted_sum", "threshold_fraction")


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def infer_disruption(
    lesion_path: str | Path,
    occupancy_path: str | Path,
    method: str = "max",
) -> np.ndarray:
    """
    Compute parcel-wise cortical disruption from a cerebellar lesion mask.

    Parameters
    ----------
    lesion_path : str or Path
        Path to a binary lesion mask NIfTI in SUIT space (3D).
    occupancy_path : str or Path
        Path to a 4D pathway occupancy volume in SUIT space.
        Shape is (X, Y, Z, N_parcels) where the 4th dimension indexes
        cortical parcels and each voxel holds the probability that a
        streamline passing through it connects to that parcel.
    method : str
        Aggregation method, one of 'max', 'mean', 'weighted_sum',
        or 'threshold_fraction'.

    Returns
    -------
    disruption : np.ndarray of shape (N_parcels,)
        Disruption probability for each cortical parcel.

    Raises
    ------
    ValueError
        If method is not recognized, or if spatial alignment fails.
    FileNotFoundError
        If either input file is missing.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Choose from: {VALID_METHODS}"
        )

    # --- Load inputs ---
    logger.info("Loading lesion mask: %s", lesion_path)
    lesion_data, lesion_affine, _ = load_nifti(lesion_path, dtype=np.float32)

    logger.info("Loading pathway occupancy: %s", occupancy_path)
    occ_data, occ_affine, _ = load_nifti(occupancy_path, dtype=np.float32)

    # --- Verify spatial alignment ---
    if not check_affine_match(lesion_affine, occ_affine):
        raise ValueError(
            "Affine mismatch between lesion mask and occupancy volume. "
            "Both must be in the same SUIT space. "
            f"Lesion affine:\n{lesion_affine}\n"
            f"Occupancy affine:\n{occ_affine}"
        )

    if not check_shape_match(lesion_data.shape, occ_data.shape):
        raise ValueError(
            "Shape mismatch between lesion mask and occupancy volume. "
            f"Lesion shape: {lesion_data.shape[:3]}, "
            f"Occupancy shape: {occ_data.shape[:3]}"
        )

    n_parcels = occ_data.shape[3]
    logger.info(
        "Occupancy volume has %d parcels, spatial shape %s",
        n_parcels, occ_data.shape[:3],
    )

    # --- Identify lesion voxels ---
    lesion_mask = lesion_data > 0
    n_lesion_voxels = int(lesion_mask.sum())
    logger.info("Lesion voxels: %d", n_lesion_voxels)

    if n_lesion_voxels == 0:
        logger.warning("Lesion mask is empty; returning zero disruption.")
        return np.zeros(n_parcels, dtype=np.float32)

    # Extract occupancy at lesion voxels: shape (N_lesion_voxels, N_parcels)
    occ_at_lesion = occ_data[lesion_mask]

    # --- Aggregate ---
    if method == "max":
        disruption = occ_at_lesion.max(axis=0)

    elif method == "mean":
        disruption = occ_at_lesion.mean(axis=0)

    elif method == "weighted_sum":
        raw_sum = occ_at_lesion.sum(axis=0)
        max_val = raw_sum.max()
        if max_val > 0:
            disruption = raw_sum / max_val
        else:
            disruption = raw_sum

    elif method == "threshold_fraction":
        # For each parcel, find the total number of voxels in the full
        # volume where occupancy exceeds the threshold (the pathway volume).
        # Then count how many of those voxels fall inside the lesion.
        above_threshold = occ_data > _THRESHOLD_FRACTION_CUTOFF
        pathway_volume = above_threshold.sum(axis=(0, 1, 2)).astype(np.float64)
        lesion_intersect = above_threshold[lesion_mask].sum(axis=0).astype(np.float64)
        # Avoid division by zero for parcels with no supra-threshold voxels
        disruption = np.where(
            pathway_volume > 0,
            lesion_intersect / pathway_volume,
            0.0,
        ).astype(np.float32)

    logger.info(
        "Disruption (%s): min=%.4f, max=%.4f, mean=%.4f",
        method, disruption.min(), disruption.max(), disruption.mean(),
    )

    return disruption.astype(np.float32)


# ---------------------------------------------------------------------------
# Volume reconstruction
# ---------------------------------------------------------------------------

def disruption_to_volume(
    disruption: np.ndarray,
    parcellation_path: str | Path,
    reference_path: str | Path,
) -> nib.Nifti1Image:
    """
    Convert a parcel-wise disruption vector to a 3D NIfTI volume.

    Each cortical voxel is assigned the disruption value of its parcel.
    Non-cortical voxels remain zero.

    Parameters
    ----------
    disruption : np.ndarray of shape (N_parcels,)
        Disruption probability per parcel (e.g., output of infer_disruption).
    parcellation_path : str or Path
        Path to a 3D integer parcellation NIfTI in SUIT space.
        Voxel values are 1-based parcel indices; 0 = background.
    reference_path : str or Path
        Path to a reference NIfTI for the output affine and header.

    Returns
    -------
    nib.Nifti1Image
        3D volume with disruption values painted onto parcels.
    """
    logger.info("Loading parcellation: %s", parcellation_path)
    parc_data, parc_affine, _ = load_nifti(parcellation_path, dtype=int)

    logger.info("Loading reference: %s", reference_path)
    _, ref_affine, ref_header = load_nifti(reference_path)

    n_parcels = len(disruption)

    # Build lookup table: index 0 = background (0.0), indices 1..N = disruption
    lut = np.zeros(n_parcels + 1, dtype=np.float32)
    lut[1:] = disruption

    # Clip parcel indices to valid range
    safe_parc = np.clip(parc_data, 0, n_parcels)
    vol = lut[safe_parc]

    logger.info(
        "Disruption volume: shape=%s, nonzero voxels=%d",
        vol.shape, int((vol > 0).sum()),
    )

    return nib.Nifti1Image(vol, ref_affine, ref_header)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summarize_disruption(
    disruption: np.ndarray,
    metadata_path: str | Path,
) -> pd.DataFrame:
    """
    Pair disruption values with parcel metadata and return a sorted DataFrame.

    Parameters
    ----------
    disruption : np.ndarray of shape (N_parcels,)
        Disruption probability per parcel.
    metadata_path : str or Path
        Path to a JSON file containing parcel metadata.
        Expected to have a top-level key 'parcels' which is a list of dicts,
        each with at least 'name' and 'nucleus_target' fields.

    Returns
    -------
    pd.DataFrame
        Columns: parcel_name, disruption, nucleus_target.
        Sorted by disruption in descending order.
    """
    logger.info("Loading parcel metadata: %s", metadata_path)
    meta = load_metadata(metadata_path)

    parcels = meta.get("parcels", [])
    if len(parcels) != len(disruption):
        logger.warning(
            "Parcel count mismatch: metadata has %d parcels, "
            "disruption array has %d entries. Truncating to the shorter.",
            len(parcels), len(disruption),
        )
        n = min(len(parcels), len(disruption))
        parcels = parcels[:n]
        disruption = disruption[:n]

    rows = []
    for i, parcel in enumerate(parcels):
        rows.append({
            "parcel_name": parcel.get("name", f"parcel_{i + 1}"),
            "disruption": float(disruption[i]),
            "nucleus_target": parcel.get("nucleus_target", "unknown"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("disruption", ascending=False).reset_index(drop=True)

    logger.info(
        "Summary: %d parcels, top disruption = %.4f (%s)",
        len(df),
        df["disruption"].iloc[0] if len(df) > 0 else 0.0,
        df["parcel_name"].iloc[0] if len(df) > 0 else "N/A",
    )

    return df


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_inference(
    lesion_path: str | Path,
    output_dir: str | Path | None = None,
    methods: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Run lesion inference with one or more aggregation methods.

    Loads the lesion mask, computes disruption for each method, optionally
    saves disruption volumes and a summary CSV.

    Parameters
    ----------
    lesion_path : str or Path
        Path to a binary lesion mask NIfTI in SUIT space.
    output_dir : str or Path, optional
        Directory for saving outputs. If None, results are returned but
        not saved to disk.
    methods : list of str, optional
        Aggregation methods to run. Defaults to all available methods.

    Returns
    -------
    dict mapping method name (str) to disruption array (np.ndarray of shape
    (N_parcels,)).
    """
    lesion_path = Path(lesion_path)
    if methods is None:
        methods = list(VALID_METHODS)

    # Validate requested methods
    for m in methods:
        if m not in VALID_METHODS:
            raise ValueError(
                f"Unknown method '{m}'. Choose from: {VALID_METHODS}"
            )

    occupancy_path = DEFAULT_OCCUPANCY_PATH
    parcellation_path = DEFAULT_PARCELLATION_PATH
    reference_path = DEFAULT_REFERENCE_PATH
    metadata_path = DEFAULT_METADATA_PATH

    logger.info("=" * 60)
    logger.info("Cerebellar Disconnectome Inference")
    logger.info("=" * 60)
    logger.info("Lesion mask:   %s", lesion_path)
    logger.info("Occupancy:     %s", occupancy_path)
    logger.info("Parcellation:  %s", parcellation_path)
    logger.info("Methods:       %s", methods)

    results = {}

    for method in methods:
        logger.info("-" * 40)
        logger.info("Running method: %s", method)
        disruption = infer_disruption(lesion_path, occupancy_path, method=method)
        results[method] = disruption

    # --- Save outputs if requested ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        lesion_stem = lesion_path.stem.replace(".nii", "")

        for method, disruption in results.items():
            # Save disruption volume
            vol_path = output_dir / f"{lesion_stem}_disruption_{method}.nii.gz"
            try:
                img = disruption_to_volume(
                    disruption, parcellation_path, reference_path,
                )
                nib.save(img, str(vol_path))
                logger.info("Saved disruption volume: %s", vol_path)
            except FileNotFoundError as exc:
                logger.warning(
                    "Could not save disruption volume for '%s': %s", method, exc,
                )

        # Save summary CSV (use the first method for the summary, then add all)
        try:
            summary_frames = []
            for method, disruption in results.items():
                df = summarize_disruption(disruption, metadata_path)
                df = df.rename(columns={"disruption": f"disruption_{method}"})
                if not summary_frames:
                    summary_frames.append(df)
                else:
                    summary_frames.append(df[[f"disruption_{method}"]])

            if summary_frames:
                summary_df = pd.concat(summary_frames, axis=1)
                csv_path = output_dir / f"{lesion_stem}_disruption_summary.csv"
                summary_df.to_csv(csv_path, index=False, float_format="%.6f")
                logger.info("Saved summary CSV: %s", csv_path)
        except FileNotFoundError as exc:
            logger.warning("Could not save summary CSV: %s", exc)

    logger.info("=" * 60)
    logger.info("Inference complete.")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.inference <lesion_SUIT.nii.gz> [output_dir]")
        sys.exit(1)

    lesion_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    run_inference(lesion_path, output_dir)
