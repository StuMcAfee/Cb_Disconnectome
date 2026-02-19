"""
Build the 4D pathway occupancy volume (Step 3 — core integration).

Combines the cortical parcellation (Step 1) with the efferent pathway maps
(Step 2) to create the 4D pathway occupancy volume that is the main data
product of the cerebellar disconnectome model.

Pipeline
--------
1. Load the native SUIT lobular parcellation (28 cortical labels at 1 mm
   isotropic resolution).
2. Load the cortico-nuclear probability map (from Step 1) and compute each
   parcel's average nuclear-target probability vector.
3. Load the nuclear efferent density maps (from Step 2).
4. For each parcel k (one per SUIT atlas label, 28 total):
       a. Weight the nuclear efferent density maps by parcel k's projection
          probabilities.
       b. Add the cortical voxels of parcel k as a "direct injury" zone
          (probability = 1.0).
       c. Store the resulting 3D map as volume k of the 4D output.

Output shape: (X, Y, Z, 28)
Each 3D volume represents the probability that a lesion at voxel (x, y, z)
disrupts cortical parcel k.  Resolution matches the native SUIT atlas (1 mm
isotropic).

Outputs
-------
- data/final/pathway_occupancy_4d.nii.gz       — main 4D volume
- data/final/pathway_occupancy_metadata.json    — JSON metadata sidecar

Usage:
    python -m src.build_4d_nifti [--config config.json]
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from src.utils import (
    ATLAS_DIR,
    DATA_FINAL,
    DATA_INTERIM,
    get_config,
    load_nifti,
    save_metadata,
    save_nifti,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Nucleus indices — must match the ordering used by cortico_nuclear_map.py
NUCLEUS_NAMES = ["fastigial", "emboliform", "globose", "dentate"]
N_NUCLEI = len(NUCLEUS_NAMES)

# Default label-to-name mapping for the 28 cortical labels in the SUIT
# anatomical atlas.  Labels 29-34 are deep nuclei and are excluded.
# Keys are the integer labels in the atlas NIfTI; values are (lobule, side)
# tuples where side is 'L', 'R', or 'V' (vermis).
DEFAULT_SUIT_LABELS: dict[int, tuple[str, str]] = {
    1:  ("I-IV",   "L"),
    2:  ("I-IV",   "R"),
    3:  ("V",      "L"),
    4:  ("V",      "R"),
    5:  ("VI",     "L"),
    6:  ("VI",     "V"),
    7:  ("VI",     "R"),
    8:  ("CrusI",  "L"),
    9:  ("CrusI",  "V"),
    10: ("CrusI",  "R"),
    11: ("CrusII", "L"),
    12: ("CrusII", "V"),
    13: ("CrusII", "R"),
    14: ("VIIb",   "L"),
    15: ("VIIb",   "V"),
    16: ("VIIb",   "R"),
    17: ("VIIIa",  "L"),
    18: ("VIIIa",  "V"),
    19: ("VIIIa",  "R"),
    20: ("VIIIb",  "L"),
    21: ("VIIIb",  "V"),
    22: ("VIIIb",  "R"),
    23: ("IX",     "L"),
    24: ("IX",     "V"),
    25: ("IX",     "R"),
    26: ("X",      "L"),
    27: ("X",      "V"),
    28: ("X",      "R"),
}

N_CORTICAL_PARCELS = len(DEFAULT_SUIT_LABELS)  # 28


# ---------------------------------------------------------------------------
# SUIT parcellation loader
# ---------------------------------------------------------------------------

def _load_suit_parcellation(config: dict) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    Locate and load the SUIT anatomical lobular parcellation NIfTI.

    Returns (data, affine, header).
    """
    candidates = list(ATLAS_DIR.rglob("*Anatom*space-SUIT*dseg.nii*"))
    if not candidates:
        raise FileNotFoundError(
            f"SUIT anatomical parcellation not found in {ATLAS_DIR}. "
            "Run 'python -m src.download' first."
        )
    parc_path = candidates[0]
    logger.info("Loading SUIT parcellation: %s", parc_path)
    return load_nifti(parc_path, dtype=int)


def load_parcellation(
    config: dict | None = None,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """
    Load the native SUIT lobular parcellation and build a parcel info list.

    Uses the 28 cortical labels from the SUIT anatomical atlas directly,
    preserving the native 1 mm isotropic resolution.

    Parameters
    ----------
    config : dict, optional
        Model configuration overrides.

    Returns
    -------
    parcellation_data : np.ndarray of shape (X, Y, Z), dtype int
        Integer label volume (labels 1-28 for cortex, 0 for background).
        Deep nuclei labels (29-34) are set to 0.
    parcel_info_list : list of dict
        One entry per cortical parcel (28 entries) with keys:
        'index'      — integer label in parcellation_data (1-28)
        'name'       — human-readable parcel name (e.g. "Left_CrusI")
        'lobule'     — SUIT lobule name
        'hemisphere' — 'L', 'R', or 'V'
    affine : np.ndarray of shape (4, 4)
        Affine matrix of the parcellation volume.
    """
    cfg = get_config(config)
    parc_data, affine, header = _load_suit_parcellation(cfg)

    # Zero out deep nuclei labels (29-34) so only cortical labels remain
    parc_data[parc_data > N_CORTICAL_PARCELS] = 0

    parcel_info_list: list[dict] = []
    for lbl_int, (lobule, side) in sorted(DEFAULT_SUIT_LABELS.items()):
        side_prefix = {"L": "Left", "R": "Right", "V": "Vermis"}[side]
        parcel_name = f"{side_prefix}_{lobule}"
        n_voxels = int((parc_data == lbl_int).sum())

        parcel_info_list.append({
            "index": lbl_int,
            "name": parcel_name,
            "lobule": lobule,
            "hemisphere": side,
        })
        logger.info(
            "  Parcel %2d: %-20s  (%d voxels)", lbl_int, parcel_name, n_voxels,
        )

    logger.info(
        "Loaded SUIT parcellation: %d cortical parcels at %s resolution.",
        len(parcel_info_list), parc_data.shape[:3],
    )

    return parc_data, parcel_info_list, affine


# ---------------------------------------------------------------------------
# Nuclear projection probabilities per parcel
# ---------------------------------------------------------------------------

def compute_parcel_nuclear_probs(
    parcellation_data: np.ndarray,
    cortico_nuclear_map: np.ndarray,
) -> np.ndarray:
    """
    For each parcel, compute the average cortico-nuclear probability vector.

    Parameters
    ----------
    parcellation_data : np.ndarray of shape (X, Y, Z)
        Integer-labelled subdivided parcellation (0 = background).
    cortico_nuclear_map : np.ndarray of shape (X, Y, Z, 4)
        Voxel-wise probabilities of projection to each of the four nuclei,
        as produced by ``src.cortico_nuclear_map``.

    Returns
    -------
    nuclear_probs : np.ndarray of shape (N_parcels, 4)
        Row k is the mean nuclear-target probability vector for parcel k+1.
        Rows are normalised to sum to 1.  Parcels with no cortical voxels
        in the cortico-nuclear map get a uniform distribution.
    """
    parcel_labels = np.unique(parcellation_data)
    parcel_labels = parcel_labels[parcel_labels > 0]
    n_parcels = len(parcel_labels)

    nuclear_probs = np.zeros((n_parcels, N_NUCLEI), dtype=np.float64)

    for i, lbl in enumerate(parcel_labels):
        mask = parcellation_data == lbl
        voxel_probs = cortico_nuclear_map[mask]  # (N_voxels, 4)

        if voxel_probs.shape[0] == 0 or voxel_probs.sum() == 0:
            # No cortical overlap — assign uniform distribution
            nuclear_probs[i] = 1.0 / N_NUCLEI
            logger.warning(
                "Parcel %d has no cortical overlap in cortico-nuclear map; "
                "using uniform nuclear distribution.", lbl,
            )
            continue

        mean_prob = voxel_probs.mean(axis=0)
        total = mean_prob.sum()
        if total > 0:
            mean_prob /= total
        else:
            mean_prob[:] = 1.0 / N_NUCLEI

        nuclear_probs[i] = mean_prob

    logger.info(
        "Computed nuclear probabilities for %d parcels.", n_parcels,
    )
    return nuclear_probs.astype(np.float32)


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------

def save_parcellation_metadata(
    parcel_info: list[dict],
    nuclear_probs: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save the JSON metadata sidecar describing the 4D pathway occupancy volume.

    Parameters
    ----------
    parcel_info : list of dict
        As returned by ``load_parcellation``.
    nuclear_probs : np.ndarray of shape (N_parcels, 4)
        Nuclear target probability vectors for each parcel.
    output_path : str or Path
        Where to write the JSON file.
    """
    parcels_meta = []
    for i, info in enumerate(parcel_info):
        probs = nuclear_probs[i]
        primary_idx = int(np.argmax(probs))
        parcels_meta.append({
            "index": info["index"],
            "name": info["name"],
            "lobule": info["lobule"],
            "hemisphere": info["hemisphere"],
            "nucleus_target": NUCLEUS_NAMES[primary_idx],
            "projection_prob": {
                NUCLEUS_NAMES[j]: round(float(probs[j]), 4)
                for j in range(N_NUCLEI)
            },
        })

    metadata = {
        "description": (
            "4D pathway occupancy volume for the cerebellar efferent "
            "disconnectome model. Each 3D sub-volume represents the "
            "probability that a lesion at voxel (x,y,z) disrupts the "
            "efferent output of the corresponding cortical parcel."
        ),
        "space": "SUIT",
        "resolution": "1 mm isotropic (native SUIT atlas)",
        "dimensions": {
            "spatial": "X, Y, Z in SUIT template space",
            "fourth": (
                f"{N_CORTICAL_PARCELS} volumes — one per native SUIT "
                "lobular label (labels 1-28)"
            ),
        },
        "parcels": parcels_meta,
        "assumptions_reference": "docs/assumptions.md",
        "normative_data_source": (
            "Elias et al. (2024) normative structural connectome, "
            "HCP multi-shell diffusion MRI. "
            "DOI: 10.1038/s41597-024-03197-0"
        ),
        "creation_date": datetime.now(timezone.utc).isoformat(),
        "software_versions": {
            "pipeline": "Cb_Disconnectome v0.1.0",
            "nibabel": nib.__version__,
            "numpy": np.__version__,
        },
    }

    save_metadata(metadata, output_path)
    logger.info("Saved parcellation metadata: %s", output_path)


# ---------------------------------------------------------------------------
# Efferent density loading helpers
# ---------------------------------------------------------------------------

def _load_cortico_nuclear_map() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the cortico-nuclear probability map produced by Step 1.

    Returns (data, affine) where data has shape (X, Y, Z, 4).
    """
    cn_path = DATA_INTERIM / "cortico_nuclear_prob_map.nii.gz"
    if not cn_path.exists():
        raise FileNotFoundError(
            f"Cortico-nuclear map not found: {cn_path}. "
            "Run 'python -m src.cortico_nuclear_map' first (Step 1)."
        )
    data, affine, _ = load_nifti(cn_path, dtype=np.float32)
    logger.info("Loaded cortico-nuclear map: %s  shape=%s", cn_path, data.shape)
    return data, affine


def _load_efferent_density_maps() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the nuclear efferent density maps produced by Step 2.

    Expects a 4D NIfTI in data/interim/ with shape (X, Y, Z, 4) where the
    4th dimension corresponds to the four nuclei (fastigial, emboliform,
    globose, dentate) in the same order as NUCLEUS_NAMES.

    Returns (data, affine).
    """
    # Search for the efferent density maps
    candidate_names = [
        "efferent_density_maps.nii.gz",
        "nuclear_efferent_density.nii.gz",
        "efferent_density_4d.nii.gz",
    ]
    efferent_path = None
    for name in candidate_names:
        p = DATA_INTERIM / name
        if p.exists():
            efferent_path = p
            break

    if efferent_path is None:
        # Try glob as a fallback
        candidates = list(DATA_INTERIM.glob("*efferent*density*.nii*"))
        if candidates:
            efferent_path = candidates[0]

    if efferent_path is None:
        raise FileNotFoundError(
            "Nuclear efferent density maps not found in "
            f"{DATA_INTERIM}. Run Step 2 (efferent pathway mapping) first."
        )

    data, affine, _ = load_nifti(efferent_path, dtype=np.float32)

    if data.ndim != 4 or data.shape[3] != N_NUCLEI:
        raise ValueError(
            f"Expected efferent density maps with shape (X, Y, Z, {N_NUCLEI}), "
            f"got {data.shape}."
        )

    logger.info(
        "Loaded efferent density maps: %s  shape=%s", efferent_path, data.shape,
    )
    return data, affine


# ---------------------------------------------------------------------------
# Main 4D build
# ---------------------------------------------------------------------------

def build_pathway_occupancy_4d(
    config: dict | None = None,
) -> Path:
    """
    Build and save the full 4D pathway occupancy volume.

    Uses the 28 native SUIT lobular labels as parcels (1 mm isotropic).
    For each cortical parcel k it:
      1. Looks up which nuclei parcel k projects to (from the cortico-nuclear
         map computed in Step 1).
      2. Weights the nuclear efferent density maps (from Step 2) by parcel k's
         projection probabilities.
      3. Adds the cortical voxels of parcel k as a "direct injury" zone
         (probability = 1.0).
      4. Stores the resulting 3D map as volume k of the 4D output.

    Parameters
    ----------
    config : dict, optional
        Model configuration overrides.

    Returns
    -------
    Path to the saved 4D NIfTI.
    """
    cfg = get_config(config)

    # ------------------------------------------------------------------
    # Step A: Load the native SUIT parcellation (28 cortical labels)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step A: Loading SUIT parcellation (28 cortical parcels)")
    logger.info("=" * 60)
    parcellation_data, parcel_info, parc_affine = load_parcellation(cfg)

    n_parcels = len(parcel_info)
    if n_parcels == 0:
        raise RuntimeError("No parcels were loaded.  Check atlas data.")

    # ------------------------------------------------------------------
    # Step B: Load the cortico-nuclear probability map (Step 1 output)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step B: Loading cortico-nuclear probability map")
    logger.info("=" * 60)
    cn_map, cn_affine = _load_cortico_nuclear_map()

    # ------------------------------------------------------------------
    # Step C: Compute per-parcel nuclear projection probabilities
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step C: Computing per-parcel nuclear probabilities")
    logger.info("=" * 60)
    nuclear_probs = compute_parcel_nuclear_probs(parcellation_data, cn_map)
    # nuclear_probs shape: (n_parcels, 4)

    # ------------------------------------------------------------------
    # Step D: Load nuclear efferent density maps (Step 2 output)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step D: Loading efferent density maps")
    logger.info("=" * 60)
    efferent_maps, eff_affine = _load_efferent_density_maps()

    # Verify spatial alignment between parcellation and efferent maps
    spatial_shape = parcellation_data.shape[:3]
    if efferent_maps.shape[:3] != spatial_shape:
        raise ValueError(
            f"Spatial shape mismatch: parcellation {spatial_shape} vs "
            f"efferent maps {efferent_maps.shape[:3]}. Both must be in "
            "SUIT space with identical resolution."
        )

    # ------------------------------------------------------------------
    # Step E: Build the 4D pathway occupancy volume
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step E: Building 4D pathway occupancy volume (%d parcels)", n_parcels)
    logger.info("=" * 60)

    density_threshold = cfg.get("EFFERENT_DENSITY_THRESHOLD", 0.01)
    occupancy_4d = np.zeros((*spatial_shape, n_parcels), dtype=np.float32)

    for i, info in enumerate(parcel_info):
        parcel_label = info["index"]
        parcel_name = info["name"]
        parcel_mask = parcellation_data == parcel_label

        logger.info(
            "  Parcel %3d / %d: %-20s  (%d cortical voxels)",
            i + 1, n_parcels, parcel_name, int(parcel_mask.sum()),
        )

        # --- Efferent pathway contribution ---
        # Weight each nucleus's efferent density map by the parcel's
        # projection probability to that nucleus, then sum across nuclei.
        prob_vec = nuclear_probs[i]  # shape (4,)

        # Efficient weighted sum: efferent_maps is (X, Y, Z, 4), prob_vec is (4,)
        # Result is (X, Y, Z) — the combined efferent density for this parcel.
        weighted_efferent = np.einsum("...n,n->...", efferent_maps, prob_vec)

        # --- Direct injury zone ---
        # Cortical voxels belonging to this parcel are always disrupted by
        # a lesion at that location (probability = 1.0).
        weighted_efferent[parcel_mask] = 1.0

        # --- Threshold very low values to zero for sparsity / file size ---
        weighted_efferent[weighted_efferent < density_threshold] = 0.0

        # --- Clip to [0, 1] ---
        np.clip(weighted_efferent, 0.0, 1.0, out=weighted_efferent)

        occupancy_4d[..., i] = weighted_efferent

    occupancy_4d = occupancy_4d.astype(np.float32)

    # ------------------------------------------------------------------
    # Step F: Save outputs
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step F: Saving outputs")
    logger.info("=" * 60)

    # 4D volume
    out_4d_path = DATA_FINAL / "pathway_occupancy_4d.nii.gz"
    save_nifti(occupancy_4d, eff_affine, out_4d_path)
    logger.info(
        "Saved 4D pathway occupancy: %s  shape=%s  dtype=%s",
        out_4d_path, occupancy_4d.shape, occupancy_4d.dtype,
    )

    # Metadata sidecar
    meta_path = DATA_FINAL / "pathway_occupancy_metadata.json"
    save_parcellation_metadata(parcel_info, nuclear_probs, meta_path)

    # Summary statistics
    nonzero_fraction = (occupancy_4d > 0).mean()
    logger.info("4D volume sparsity: %.1f%% nonzero", nonzero_fraction * 100)
    logger.info(
        "Value range: [%.4f, %.4f]",
        float(occupancy_4d.min()), float(occupancy_4d.max()),
    )
    logger.info("Build complete.")

    return out_4d_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the 4D pathway occupancy volume (Step 3 — core integration)"
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file with parameter overrides",
    )
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    build_pathway_occupancy_4d(config=config)


if __name__ == "__main__":
    main()
