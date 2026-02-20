"""
Cortico-nuclear projection map builder (Step 1).

Creates a 3D volume in SUIT space where each voxel is labeled with its target
deep cerebellar nucleus (or nuclei, with probabilities).

Output: a 4D NIfTI in SUIT space with dimensions (X, Y, Z, 8), where the
8th dimension represents probability of projection to bilateral nuclei:
  [0] = left fastigial        [4] = right fastigial
  [1] = left emboliform       [5] = right emboliform
  [2] = left globose           [6] = right globose
  [3] = left dentate           [7] = right dentate

For each cortical voxel, the probabilities sum to 1.0.
For non-cortex voxels (white matter, nuclei, outside cerebellum), all
probabilities are 0.

The mapping follows the Voogd zonal scheme:
  - Vermis -> fastigial (bilateral)
  - Paravermal lobules I-V -> emboliform (ipsilateral)
  - Paravermal lobules VI-IX -> globose (ipsilateral)
  - Lateral hemisphere anterior -> dentate (ipsilateral, dorsal/motor)
  - Lateral hemisphere Crus I/II -> dentate (ipsilateral, ventral/nonmotor)
  - Lateral hemisphere VIIb-IX -> dentate (ipsilateral, intermediate)
  - Flocculonodular (X) -> fastigial + vestibular (bilateral)

Laterality is determined from the SUIT label name: labels starting with
"Left" project ipsilaterally to left-side nuclei, "Right" to right-side
nuclei, and "Vermis" labels split projections equally between both sides.

Transitional zones are handled with sigmoid blending at boundaries.

See docs/assumptions.md for the anatomical basis of this mapping.

Usage:
    python -m src.cortico_nuclear_map [--config config.json]
"""

import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.utils import (
    ATLAS_DIR,
    DATA_INTERIM,
    PROJECT_ROOT,
    DEFAULT_CONFIG,
    get_config,
    get_mm_coordinate_grid,
    load_nifti,
    save_nifti,
    classify_zones,
    sigmoid_weight,
    ensure_directories,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nucleus constants (bilateral: 4 per hemisphere = 8 total)
# ---------------------------------------------------------------------------

# Unilateral nucleus types (used for the Voogd zonal probability tables)
UNILATERAL_NUCLEUS_NAMES = ["fastigial", "emboliform", "globose", "dentate"]
N_UNILATERAL = len(UNILATERAL_NUCLEUS_NAMES)

# Bilateral nucleus channels: left (0-3), right (4-7)
NUCLEUS_NAMES = [
    "left_fastigial", "left_emboliform", "left_globose", "left_dentate",
    "right_fastigial", "right_emboliform", "right_globose", "right_dentate",
]
N_NUCLEI = len(NUCLEUS_NAMES)

NUCLEUS_INDEX = {name: i for i, name in enumerate(NUCLEUS_NAMES)}

# ---------------------------------------------------------------------------
# Label laterality mapping
# ---------------------------------------------------------------------------
# Determined from SUIT atlas label names:
#   "Left *"   -> "left"   (projects to left-side nuclei, indices 0-3)
#   "Right *"  -> "right"  (projects to right-side nuclei, indices 4-7)
#   "Vermis *" -> "vermis" (projects bilaterally, split 50/50)
#
# Labels 1-4 (I-IV, V) have no separate vermis label in SUIT; they are
# assigned as left/right based on their label name.  Their midline voxels
# receive high vermis zone weight via the sigmoid blending, which correctly
# assigns them mostly to fastigial regardless of side.

LABEL_LATERALITY = {
    1: "left",    2: "right",                         # I-IV
    3: "left",    4: "right",                         # V
    5: "left",    6: "vermis",   7: "right",          # VI
    8: "left",    9: "vermis",  10: "right",          # CrusI
   11: "left",   12: "vermis",  13: "right",          # CrusII
   14: "left",   15: "vermis",  16: "right",          # VIIb
   17: "left",   18: "vermis",  19: "right",          # VIIIa
   20: "left",   21: "vermis",  22: "right",          # VIIIb
   23: "left",   24: "vermis",  25: "right",          # IX
   26: "left",   27: "vermis",  28: "right",          # X
   # Deep nuclei — not cortical, excluded from projection mapping
   29: None, 30: None, 31: None, 32: None, 33: None, 34: None,
}


def _expand_bilateral(
    unilateral_probs: np.ndarray,
    laterality: str,
) -> np.ndarray:
    """
    Expand a 4-element unilateral probability vector to 8-element bilateral.

    Parameters
    ----------
    unilateral_probs : np.ndarray of shape (4,)
        Probability vector [fastigial, emboliform, globose, dentate].
    laterality : str
        One of 'left', 'right', or 'vermis'.

    Returns
    -------
    bilateral_probs : np.ndarray of shape (8,)
        [left_F, left_E, left_G, left_D, right_F, right_E, right_G, right_D]
    """
    bilateral = np.zeros(N_NUCLEI, dtype=np.float32)
    if laterality == "left":
        bilateral[:N_UNILATERAL] = unilateral_probs
    elif laterality == "right":
        bilateral[N_UNILATERAL:] = unilateral_probs
    elif laterality == "vermis":
        bilateral[:N_UNILATERAL] = unilateral_probs * 0.5
        bilateral[N_UNILATERAL:] = unilateral_probs * 0.5
    return bilateral

# ---------------------------------------------------------------------------
# Lobule-to-zone mapping table (Assumption A1)
# ---------------------------------------------------------------------------
# Maps SUIT atlas integer labels to an anterior/posterior group classification.
#
# Integer label -> classification string:
#   "anterior"           : lobules I-V
#   "superior_posterior"  : lobules VI, Crus I, Crus II
#   "inferior_posterior"  : lobules VIIb, VIII, IX
#   "flocculonodular"     : lobule X
#
# Labels for deep nuclei (Dentate, Interposed, Fastigial) are mapped to None
# since they are not cortical voxels and receive no projection probabilities.
#
# The default SUIT anatomical parcellation uses labels 1-34 as described in
# the Diedrichsen cerebellar atlas repository.

LOBULE_ZONE_MAP = {
    # Lobules I-IV (anterior)
    1:  "anterior",      # Left I-IV
    2:  "anterior",      # Right I-IV
    # Lobule V (anterior)
    3:  "anterior",      # Left V
    4:  "anterior",      # Right V
    # Lobule VI (superior posterior)
    5:  "superior_posterior",   # Left VI
    6:  "superior_posterior",   # Vermis VI
    7:  "superior_posterior",   # Right VI
    # Crus I (superior posterior)
    8:  "superior_posterior",   # Left Crus I
    9:  "superior_posterior",   # Vermis Crus I
    10: "superior_posterior",   # Right Crus I
    # Crus II (superior posterior)
    11: "superior_posterior",   # Left Crus II
    12: "superior_posterior",   # Vermis Crus II
    13: "superior_posterior",   # Right Crus II
    # Lobule VIIb (inferior posterior)
    14: "inferior_posterior",   # Left VIIb
    15: "inferior_posterior",   # Vermis VIIb
    16: "inferior_posterior",   # Right VIIb
    # Lobule VIIIa (inferior posterior)
    17: "inferior_posterior",   # Left VIIIa
    18: "inferior_posterior",   # Vermis VIIIa
    19: "inferior_posterior",   # Right VIIIa
    # Lobule VIIIb (inferior posterior)
    20: "inferior_posterior",   # Left VIIIb
    21: "inferior_posterior",   # Vermis VIIIb
    22: "inferior_posterior",   # Right VIIIb
    # Lobule IX (inferior posterior)
    23: "inferior_posterior",   # Left IX
    24: "inferior_posterior",   # Vermis IX
    25: "inferior_posterior",   # Right IX
    # Lobule X (flocculonodular)
    26: "flocculonodular",     # Left X
    27: "flocculonodular",     # Vermis X
    28: "flocculonodular",     # Right X
    # Deep nuclei -- not cortical, excluded from mapping
    29: None,   # Left Dentate
    30: None,   # Right Dentate
    31: None,   # Left Interposed
    32: None,   # Right Interposed
    33: None,   # Left Fastigial
    34: None,   # Right Fastigial
}

# ---------------------------------------------------------------------------
# Lobule group -> per-zone nuclear projection probabilities
# ---------------------------------------------------------------------------
# Each entry maps a lobule group to nuclear target probabilities per zone.
# Format: {zone: [fastigial, emboliform, globose, dentate]}
#
# These probability vectors encode the Voogd zonal scheme (Assumption A1):
#   - Vermis projects predominantly to fastigial
#   - Paravermis: anterior lobules -> emboliform; posterior -> globose
#   - Lateral hemisphere -> dentate (with subregional variation)
#   - Flocculonodular lobe -> fastigial (+ vestibular, not modeled here)

LOBULE_GROUP_PROBS = {
    "anterior": {
        # Lobules I-V
        "vermis":      [0.90, 0.05, 0.03, 0.02],
        "paravermis":  [0.05, 0.70, 0.20, 0.05],
        "lateral":     [0.02, 0.08, 0.05, 0.85],  # dentate dorsal/motor
    },
    "superior_posterior": {
        # Lobules VI, Crus I, Crus II
        "vermis":      [0.85, 0.05, 0.08, 0.02],
        "paravermis":  [0.05, 0.15, 0.65, 0.15],
        "lateral":     [0.02, 0.03, 0.05, 0.90],  # dentate ventral/nonmotor
    },
    "inferior_posterior": {
        # Lobules VIIb, VIII, IX
        "vermis":      [0.85, 0.05, 0.08, 0.02],
        "paravermis":  [0.05, 0.20, 0.60, 0.15],
        "lateral":     [0.02, 0.05, 0.08, 0.85],  # dentate intermediate
    },
    "flocculonodular": {
        # Lobule X (flocculonodular lobe)
        # Strong fastigial projection across all zones; vestibular nuclei
        # are not modeled as a separate output channel, so their contribution
        # is absorbed into the fastigial column.
        "vermis":      [0.92, 0.03, 0.03, 0.02],
        "paravermis":  [0.80, 0.10, 0.05, 0.05],
        "lateral":     [0.60, 0.10, 0.10, 0.20],
    },
}


# ---------------------------------------------------------------------------
# Lobule label loading
# ---------------------------------------------------------------------------

def get_lobule_label_map(atlas_dir: Path | None = None) -> dict[int, str]:
    """
    Load the integer-label-to-name mapping for the SUIT anatomical atlas.

    First searches for the accompanying .tsv file that ships with the
    Diedrichsen cerebellar_atlases repository.  If no .tsv is found,
    falls back to a hardcoded default mapping that matches the standard
    SUIT label ordering.

    Parameters
    ----------
    atlas_dir : Path, optional
        Root directory of the cerebellar atlas data.
        Defaults to ``ATLAS_DIR`` from ``src.utils``.

    Returns
    -------
    dict
        Mapping from integer label to human-readable lobule name string.
    """
    if atlas_dir is None:
        atlas_dir = ATLAS_DIR

    # Search for the label .tsv file that accompanies the parcellation NIfTI
    tsv_candidates = list(atlas_dir.rglob("*Anatom*space-SUIT*.tsv"))
    if not tsv_candidates:
        tsv_candidates = list(atlas_dir.rglob("*Anatom*.tsv"))

    if tsv_candidates:
        tsv_path = tsv_candidates[0]
        logger.info("Loading lobule labels from: %s", tsv_path)
        df = pd.read_csv(tsv_path, sep="\t")
        # The TSV typically has columns named 'index' and 'name', but some
        # versions may use different column headers.
        idx_col = "index" if "index" in df.columns else df.columns[0]
        name_col = "name" if "name" in df.columns else df.columns[1]
        return dict(zip(df[idx_col].astype(int), df[name_col].astype(str)))

    # Fallback: hardcoded default SUIT label ordering
    logger.warning(
        "No .tsv label file found in %s. Using default SUIT label mapping.",
        atlas_dir,
    )
    default_labels = {
        1: "Left I-IV",      2: "Right I-IV",
        3: "Left V",         4: "Right V",
        5: "Left VI",        6: "Vermis VI",       7: "Right VI",
        8: "Left CrusI",     9: "Vermis CrusI",    10: "Right CrusI",
        11: "Left CrusII",   12: "Vermis CrusII",  13: "Right CrusII",
        14: "Left VIIb",     15: "Vermis VIIb",    16: "Right VIIb",
        17: "Left VIIIa",    18: "Vermis VIIIa",   19: "Right VIIIa",
        20: "Left VIIIb",    21: "Vermis VIIIb",   22: "Right VIIIb",
        23: "Left IX",       24: "Vermis IX",       25: "Right IX",
        26: "Left X",        27: "Vermis X",        28: "Right X",
        29: "Left Dentate",  30: "Right Dentate",
        31: "Left Interposed", 32: "Right Interposed",
        33: "Left Fastigial", 34: "Right Fastigial",
    }
    return default_labels


def _identify_lobule_group(label_name: str) -> str | None:
    """
    Map a SUIT atlas lobule label name to one of the lobule groups.

    The SUIT atlas uses names like 'Left I-IV', 'Right CrusI', 'Vermis VI',
    etc.  We strip the laterality prefix and match against known lobule
    patterns to determine the group classification.

    Parameters
    ----------
    label_name : str
        Human-readable lobule name from the SUIT atlas.

    Returns
    -------
    str or None
        One of 'anterior', 'superior_posterior', 'inferior_posterior',
        'flocculonodular', or None if the label is not a cortical lobule
        (e.g. deep nuclei labels).
    """
    name = label_name.strip()
    for prefix in ["Left ", "Right ", "Vermis ", "L ", "R "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.strip()

    # Anterior: I-IV, V
    anterior_patterns = ["I-IV", "V"]
    # Superior posterior: VI, CrusI, CrusII
    sup_post_patterns = ["VI", "CrusI", "CrusII", "Crus I", "Crus II"]
    # Inferior posterior: VIIb, VIIIa, VIIIb, VIII, IX
    inf_post_patterns = ["VIIb", "VIIIa", "VIIIb", "VIII", "IX"]
    # Flocculonodular: X
    flocculonodular_patterns = ["X"]

    # Check flocculonodular first (X must not match VIIIx patterns)
    if name == "X":
        return "flocculonodular"

    # Check inferior posterior before superior posterior
    # (VIIb, VIII patterns must match before VI)
    for pat in inf_post_patterns:
        if pat in name:
            return "inferior_posterior"

    for pat in sup_post_patterns:
        if pat in name:
            return "superior_posterior"

    for pat in anterior_patterns:
        if pat in name:
            return "anterior"

    # Deep nuclei (Dentate, Interposed, Fastigial) or unrecognized
    return None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_nuclear_probabilities(
    parc_data: np.ndarray,
    affine: np.ndarray,
    label_map: dict[int, str],
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-voxel nuclear projection probabilities for the cerebellar cortex.

    For each cortical voxel the function:
      1. Determines its lobular label from the parcellation.
      2. Determines its medial-lateral position in SUIT space (x-coordinate).
      3. Classifies it as vermis / paravermal / lateral hemisphere using
         sigmoid blending (``classify_zones``).
      4. Applies the Voogd-scheme mapping table to produce a probability
         vector over the four deep cerebellar nuclei.

    Transitional zones between vermis, paravermis, and lateral hemisphere are
    handled with sigmoid blending so that voxels near boundaries receive
    mixed contributions from adjacent zones.

    Parameters
    ----------
    parc_data : np.ndarray
        3D integer parcellation volume (SUIT anatomical labels).
    affine : np.ndarray of shape (4, 4)
        Affine matrix mapping voxel indices to mm coordinates.
    label_map : dict
        Mapping from integer label to lobule name string (as returned by
        ``get_lobule_label_map``).
    config : dict, optional
        Model configuration overrides (zone thresholds, transition widths).

    Returns
    -------
    prob_map : np.ndarray of shape (*parc_data.shape, 8)
        Probability of projection to each bilateral nucleus at every voxel.
        Indices 0-3 are left nuclei, 4-7 are right nuclei.
        Sums to 1.0 for cortical voxels; all zeros for non-cortical voxels.
    cortical_mask : np.ndarray of shape parc_data.shape, dtype bool
        Boolean mask identifying cortical voxels that received probabilities.
    """
    cfg = get_config(config)
    shape = parc_data.shape[:3]

    # --- Determine the lobule group for each atlas label ---
    label_to_group: dict[int, str] = {}
    cortical_labels: set[int] = set()

    for lbl_int, lbl_name in label_map.items():
        # Check LOBULE_ZONE_MAP first (integer-based lookup)
        if lbl_int in LOBULE_ZONE_MAP:
            group = LOBULE_ZONE_MAP[lbl_int]
        else:
            # Fall back to name-based matching
            group = _identify_lobule_group(lbl_name)

        if group is not None:
            label_to_group[lbl_int] = group
            cortical_labels.add(lbl_int)
            logger.debug(
                "Label %d (%s) -> group '%s'", lbl_int, lbl_name, group
            )
        else:
            logger.debug(
                "Label %d (%s) -> non-cortical (excluded)", lbl_int, lbl_name
            )

    logger.info(
        "Identified %d cortical labels across %d lobule groups.",
        len(cortical_labels),
        len(set(label_to_group.values())),
    )

    # --- Build coordinate grid and classify medial-lateral zones ---
    mm_grid = get_mm_coordinate_grid(shape, affine)
    x_mm = mm_grid[..., 0]  # medial-lateral axis
    zone_weights = classify_zones(x_mm, cfg)

    # --- Allocate output (8 bilateral channels) ---
    prob_map = np.zeros((*shape, N_NUCLEI), dtype=np.float32)

    # --- Assign nuclear target probabilities per voxel ---
    for lbl_int in sorted(cortical_labels):
        mask = parc_data == lbl_int
        n_voxels = mask.sum()
        if n_voxels == 0:
            continue

        group_key = label_to_group[lbl_int]
        group_probs = LOBULE_GROUP_PROBS[group_key]
        laterality = LABEL_LATERALITY.get(lbl_int, "left")

        for zone_name in ("vermis", "paravermis", "lateral"):
            zone_w = zone_weights[zone_name][mask]  # (N_voxels,)
            unilateral_probs = np.array(
                group_probs[zone_name], dtype=np.float32
            )  # (4,)

            # Expand to bilateral (8,) based on label laterality
            bilateral_probs = _expand_bilateral(
                unilateral_probs, laterality
            )  # (8,)

            # Each voxel receives zone_weight * bilateral_probability_vector
            contribution = zone_w[:, np.newaxis] * bilateral_probs[np.newaxis, :]
            prob_map[mask] += contribution

        logger.debug(
            "  Label %d: %d voxels, group='%s', laterality='%s'",
            lbl_int, n_voxels, group_key, laterality,
        )

    # --- Normalize probabilities to sum to 1 at each voxel ---
    prob_sum = prob_map.sum(axis=-1, keepdims=True)
    prob_sum = np.where(prob_sum > 0, prob_sum, 1.0)  # avoid division by zero
    prob_map = prob_map / prob_sum

    # --- Zero out any non-cortical voxels ---
    cortical_mask = np.isin(parc_data, list(cortical_labels))
    prob_map[~cortical_mask] = 0.0

    n_cortical = cortical_mask.sum()
    logger.info(
        "Computed nuclear probabilities for %d cortical voxels (%.1f%% of volume).",
        n_cortical,
        100.0 * n_cortical / parc_data.size,
    )

    return prob_map, cortical_mask


def create_winner_take_all_map(
    prob_map: np.ndarray,
    cortical_mask: np.ndarray,
) -> np.ndarray:
    """
    Create a winner-take-all map by selecting the nucleus with the highest
    probability at each cortical voxel.

    Parameters
    ----------
    prob_map : np.ndarray of shape (X, Y, Z, 8)
        Per-voxel bilateral nuclear projection probabilities.
    cortical_mask : np.ndarray of shape (X, Y, Z), dtype bool
        Boolean mask of cortical voxels.

    Returns
    -------
    winner_map : np.ndarray of shape (X, Y, Z), dtype float32
        Integer label of the winning nucleus at each cortical voxel
        (0-7 indexing into NUCLEUS_NAMES).
        Non-cortical voxels are set to -1.
    """
    winner_map = np.full(prob_map.shape[:3], -1.0, dtype=np.float32)
    winner_map[cortical_mask] = np.argmax(
        prob_map[cortical_mask], axis=-1
    ).astype(np.float32)

    # Log distribution of winning nuclei
    for idx, name in enumerate(NUCLEUS_NAMES):
        n_voxels = np.sum(winner_map == idx)
        logger.info("  Winner-take-all: %s = %d voxels", name, n_voxels)

    return winner_map


# ---------------------------------------------------------------------------
# Main builder function
# ---------------------------------------------------------------------------

def build_cortico_nuclear_map(
    config: dict | None = None,
    parcellation_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Build the cortico-nuclear projection probability map and save outputs.

    This is the primary entry point for Step 1 of the disconnectome pipeline.
    It loads the SUIT anatomical parcellation, computes voxel-wise nuclear
    projection probabilities based on the Voogd zonal scheme, and saves:

      1. ``cortico_nuclear_prob_map.nii.gz`` -- 4D probability map
         (X, Y, Z, 4) where the 4th dimension indexes the four deep nuclei.
      2. ``cortico_nuclear_winner.nii.gz`` -- 3D winner-take-all map
         labeling each cortical voxel with its most probable target nucleus.
      3. ``cortico_nuclear_metadata.json`` -- summary statistics and
         configuration parameters used.

    Parameters
    ----------
    config : dict, optional
        Model configuration overrides (see ``DEFAULT_CONFIG`` in src.utils).
    parcellation_path : str or Path, optional
        Path to the SUIT anatomical parcellation NIfTI.
        If None, searches ``ATLAS_DIR`` for the standard filename.
    output_dir : str or Path, optional
        Where to save outputs. Defaults to ``DATA_INTERIM``.

    Returns
    -------
    Path
        Path to the saved 4D probability map NIfTI file.
    """
    cfg = get_config(config)
    output_dir = Path(output_dir) if output_dir else DATA_INTERIM
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load the SUIT anatomical parcellation (lobular labels)
    # ------------------------------------------------------------------
    if parcellation_path is None:
        candidates = list(ATLAS_DIR.rglob("*Anatom*space-SUIT*dseg.nii*"))
        if not candidates:
            raise FileNotFoundError(
                f"SUIT anatomical parcellation not found in {ATLAS_DIR}. "
                "Run 'python -m src.download --step atlases' first."
            )
        parcellation_path = candidates[0]

    logger.info("Loading parcellation: %s", parcellation_path)
    parc_data, affine, header = load_nifti(parcellation_path, dtype=int)
    shape = parc_data.shape[:3]
    logger.info(
        "Parcellation shape: %s, unique labels: %s",
        shape,
        np.unique(parc_data).tolist(),
    )

    # ------------------------------------------------------------------
    # 2. Load lobule label names
    # ------------------------------------------------------------------
    label_map = get_lobule_label_map(ATLAS_DIR)
    logger.info("Loaded %d lobule labels.", len(label_map))

    # ------------------------------------------------------------------
    # 3. Compute nuclear projection probabilities
    # ------------------------------------------------------------------
    prob_map, cortical_mask = compute_nuclear_probabilities(
        parc_data, affine, label_map, cfg
    )

    # ------------------------------------------------------------------
    # 4. Create winner-take-all map
    # ------------------------------------------------------------------
    winner_map = create_winner_take_all_map(prob_map, cortical_mask)

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------

    # 4D probability map
    prob_path = output_dir / "cortico_nuclear_prob_map.nii.gz"
    save_nifti(prob_map, affine, prob_path)
    logger.info(
        "Cortico-nuclear probability map saved: %s  shape=%s",
        prob_path, prob_map.shape,
    )

    # 3D winner-take-all map
    winner_path = output_dir / "cortico_nuclear_winner.nii.gz"
    save_nifti(winner_map, affine, winner_path)
    logger.info("Winner-take-all map saved: %s", winner_path)

    # ------------------------------------------------------------------
    # 6. Save metadata sidecar
    # ------------------------------------------------------------------
    cortical_labels = {
        lbl for lbl, grp in LOBULE_ZONE_MAP.items() if grp is not None
    }
    parcel_meta = []
    for lbl_int in sorted(cortical_labels):
        name = label_map.get(lbl_int, f"label_{lbl_int}")
        mask = parc_data == lbl_int
        if mask.any():
            avg_prob = prob_map[mask].mean(axis=0).tolist()
        else:
            avg_prob = [0.0] * N_NUCLEI
        parcel_meta.append({
            "label": int(lbl_int),
            "name": name,
            "group": LOBULE_ZONE_MAP.get(lbl_int, "unknown"),
            "laterality": LABEL_LATERALITY.get(lbl_int, "unknown"),
            "avg_nuclear_prob": {
                NUCLEUS_NAMES[i]: round(avg_prob[i], 4)
                for i in range(N_NUCLEI)
            },
            "n_voxels": int(mask.sum()),
        })

    meta_path = output_dir / "cortico_nuclear_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "description": (
                "Cortico-nuclear projection probability map (bilateral). "
                "Indices 0-3 are left nuclei, 4-7 are right nuclei."
            ),
            "space": "SUIT",
            "n_nuclei": N_NUCLEI,
            "nucleus_names": NUCLEUS_NAMES,
            "nucleus_indices": NUCLEUS_INDEX,
            "unilateral_nucleus_names": UNILATERAL_NUCLEUS_NAMES,
            "config": cfg,
            "parcels": parcel_meta,
        }, f, indent=2)
    logger.info("Metadata saved: %s", meta_path)

    return prob_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Parse command-line arguments and run the map builder."""
    parser = argparse.ArgumentParser(
        description="Build cortico-nuclear projection map (Step 1)"
    )
    parser.add_argument(
        "--parcellation", type=str, default=None,
        help="Path to SUIT anatomical parcellation NIfTI",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file with parameter overrides",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/interim)",
    )
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    ensure_directories()
    build_cortico_nuclear_map(
        config=config,
        parcellation_path=args.parcellation,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    main()
