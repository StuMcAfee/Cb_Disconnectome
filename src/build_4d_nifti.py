"""
Build vertex-level nuclear projection data (Step 3 — core integration).

Combines the cortico-nuclear probability map (Step 1) with the SUIT surface
meshes to create per-vertex nuclear projection probabilities.  This gives
vertex-level resolution (28,935 surface vertices) instead of the former
28-parcel lobule-level approach.

Pipeline
--------
1. Load SUIT pial and white-matter surface GIfTI files (28,935 matched
   vertices).
2. Load the cortico-nuclear probability map from Step 1 (X, Y, Z, 4).
3. For each vertex, sample the cortico-nuclear map at 6 depths between the
   pial and white surfaces, then average to produce a (4,) nuclear projection
   probability vector.
4. Determine each vertex's SUIT lobular label for summary/reporting.
5. Load and combine the four per-nucleus efferent density maps (Step 2) into
   a single 4D NIfTI in SUIT space for use at inference time.

Outputs
-------
- data/final/vertex_projections.npz
      projections  (28935, 4)   nuclear projection probabilities
      pial_coords  (28935, 3)   pial surface coordinates (SUIT mm)
      white_coords (28935, 3)   white surface coordinates (SUIT mm)
      parcel_labels (28935,)    SUIT lobular label per vertex (1-28; 0=outside)
- data/final/vertex_metadata.json         — JSON metadata sidecar
- data/final/efferent_density_suit_4d.nii.gz — combined efferent density maps

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

# Depth fractions for sampling between pial and white surfaces
# (matches SUITPy vol_to_surf defaults)
DEFAULT_DEPTHS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# ---------------------------------------------------------------------------
# SUIT surface loading
# ---------------------------------------------------------------------------

def _find_suitpy_surfaces_dir() -> Path:
    """Locate the SUITPy surfaces directory."""
    try:
        import SUITPy as suit
        surfaces_dir = Path(suit.__file__).parent / "surfaces"
    except ImportError:
        # Fallback: search common locations
        fallback = Path("/tmp/suitpy_download")
        candidates = list(fallback.rglob("surfaces/PIAL_SUIT.surf.gii"))
        if candidates:
            surfaces_dir = candidates[0].parent
        else:
            raise FileNotFoundError(
                "SUITPy not installed and no surfaces found. "
                "Install with: pip install git+https://github.com/"
                "DiedrichsenLab/SUITPy.git"
            )
    if not surfaces_dir.exists():
        raise FileNotFoundError(f"SUITPy surfaces directory not found: {surfaces_dir}")
    return surfaces_dir


def load_suit_surfaces(space: str = "SUIT") -> tuple[np.ndarray, np.ndarray]:
    """
    Load matched pial and white-matter surfaces from SUITPy.

    Parameters
    ----------
    space : str
        Coordinate space: 'SUIT', 'MNISymC', 'FSL', or 'SPM'.

    Returns
    -------
    pial_coords : np.ndarray of shape (N_vertices, 3)
        Pial surface vertex coordinates in mm.
    white_coords : np.ndarray of shape (N_vertices, 3)
        White-matter surface vertex coordinates in mm.
    """
    surfaces_dir = _find_suitpy_surfaces_dir()

    pial_path = surfaces_dir / f"PIAL_{space}.surf.gii"
    white_path = surfaces_dir / f"WHITE_{space}.surf.gii"

    if not pial_path.exists():
        raise FileNotFoundError(f"Pial surface not found: {pial_path}")
    if not white_path.exists():
        raise FileNotFoundError(f"White surface not found: {white_path}")

    pial_gii = nib.load(str(pial_path))
    white_gii = nib.load(str(white_path))

    pial_coords = pial_gii.darrays[0].data.astype(np.float64)
    white_coords = white_gii.darrays[0].data.astype(np.float64)

    n_pial = pial_coords.shape[0]
    n_white = white_coords.shape[0]
    if n_pial != n_white:
        raise ValueError(
            f"Vertex count mismatch: pial has {n_pial}, white has {n_white}."
        )

    logger.info(
        "Loaded SUIT surfaces (%s space): %d vertices", space, n_pial,
    )
    return pial_coords, white_coords


# ---------------------------------------------------------------------------
# Volume sampling at vertex locations
# ---------------------------------------------------------------------------

def sample_volume_at_vertices(
    volume_data: np.ndarray,
    affine: np.ndarray,
    pial_coords: np.ndarray,
    white_coords: np.ndarray,
    depths: tuple[float, ...] = DEFAULT_DEPTHS,
) -> np.ndarray:
    """
    Sample a 3D or 4D volume at vertex locations using depth interpolation.

    For each vertex, computes 3D coordinates at multiple depths between the
    pial and white surfaces, converts to voxel indices, samples the volume,
    and averages across depths.

    Parameters
    ----------
    volume_data : np.ndarray of shape (X, Y, Z) or (X, Y, Z, C)
        Volume to sample.
    affine : np.ndarray of shape (4, 4)
        Affine mapping voxel indices to mm coordinates.
    pial_coords : np.ndarray of shape (N_vertices, 3)
        Pial surface coordinates in mm.
    white_coords : np.ndarray of shape (N_vertices, 3)
        White surface coordinates in mm.
    depths : tuple of float
        Fractional depths between pial (0) and white (1) surfaces.

    Returns
    -------
    vertex_values : np.ndarray of shape (N_vertices,) or (N_vertices, C)
        Sampled values averaged across depths.
    """
    n_vertices = pial_coords.shape[0]
    is_4d = volume_data.ndim == 4
    n_channels = volume_data.shape[3] if is_4d else 1
    spatial_shape = np.array(volume_data.shape[:3])

    inv_affine = np.linalg.inv(affine)

    accumulated = np.zeros((n_vertices, n_channels), dtype=np.float64)
    n_valid = np.zeros(n_vertices, dtype=np.int32)

    for depth in depths:
        # Interpolate between pial and white surface coordinates
        coords_mm = (1.0 - depth) * pial_coords + depth * white_coords

        # Convert mm to voxel indices
        ones = np.ones((n_vertices, 1), dtype=np.float64)
        homogeneous = np.hstack([coords_mm, ones])
        voxel_float = (inv_affine @ homogeneous.T).T[:, :3]
        voxel_idx = np.round(voxel_float).astype(int)

        # Check bounds
        in_bounds = (
            np.all(voxel_idx >= 0, axis=1)
            & (voxel_idx[:, 0] < spatial_shape[0])
            & (voxel_idx[:, 1] < spatial_shape[1])
            & (voxel_idx[:, 2] < spatial_shape[2])
        )

        valid_idx = voxel_idx[in_bounds]
        if valid_idx.shape[0] == 0:
            continue

        if is_4d:
            values = volume_data[
                valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2], :
            ]
        else:
            values = volume_data[
                valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
            ].reshape(-1, 1)

        accumulated[in_bounds] += values
        n_valid[in_bounds] += 1

    # Average across depths (avoid division by zero)
    n_valid_safe = np.maximum(n_valid, 1)
    vertex_values = (accumulated / n_valid_safe[:, np.newaxis]).astype(np.float32)

    if not is_4d:
        vertex_values = vertex_values.squeeze(axis=1)

    return vertex_values


# ---------------------------------------------------------------------------
# SUIT parcellation loader (for per-vertex lobule labels)
# ---------------------------------------------------------------------------

def _load_suit_parcellation(config: dict) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Locate and load the SUIT anatomical lobular parcellation NIfTI."""
    candidates = list(ATLAS_DIR.rglob("*Anatom*space-SUIT*dseg.nii*"))
    if not candidates:
        raise FileNotFoundError(
            f"SUIT anatomical parcellation not found in {ATLAS_DIR}. "
            "Run 'python -m src.download' first."
        )
    parc_path = candidates[0]
    logger.info("Loading SUIT parcellation: %s", parc_path)
    return load_nifti(parc_path, dtype=int)


# ---------------------------------------------------------------------------
# Cortico-nuclear map loader
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


# ---------------------------------------------------------------------------
# Efferent density map loading
# ---------------------------------------------------------------------------

def _load_efferent_density_maps() -> tuple[np.ndarray, np.ndarray]:
    """
    Load efferent density maps from Step 2.

    Looks first for a combined 4D NIfTI, then falls back to loading and
    stacking 4 individual per-nucleus files.

    Returns (data, affine) where data has shape (X, Y, Z, 4).
    """
    # Strategy 1: combined 4D file
    for name in [
        "efferent_density_maps.nii.gz",
        "efferent_density_4d.nii.gz",
        "nuclear_efferent_density.nii.gz",
    ]:
        p = DATA_INTERIM / name
        if p.exists():
            data, affine, _ = load_nifti(p, dtype=np.float32)
            if data.ndim == 4 and data.shape[3] == N_NUCLEI:
                logger.info("Loaded combined efferent maps: %s", p)
                return data, affine

    # Strategy 2: individual per-nucleus files
    individual = []
    affine = None
    for nucleus in NUCLEUS_NAMES:
        candidates = sorted(DATA_INTERIM.glob(f"*efferent*density*{nucleus}*.nii*"))
        if not candidates:
            raise FileNotFoundError(
                f"Efferent density map for '{nucleus}' not found in "
                f"{DATA_INTERIM}. Run Step 2 first."
            )
        data_n, aff_n, _ = load_nifti(candidates[0], dtype=np.float32)
        individual.append(data_n)
        if affine is None:
            affine = aff_n
        logger.info("  Loaded %s: %s", nucleus, candidates[0].name)

    data = np.stack(individual, axis=-1)
    logger.info("Stacked efferent maps: shape=%s", data.shape)
    return data, affine


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------

def _build_parcel_label_name(label_int: int) -> str:
    """Build a human-readable name for a SUIT lobular label."""
    if label_int in DEFAULT_SUIT_LABELS:
        lobule, side = DEFAULT_SUIT_LABELS[label_int]
        prefix = {"L": "Left", "R": "Right", "V": "Vermis"}[side]
        return f"{prefix}_{lobule}"
    return f"label_{label_int}"


def save_vertex_metadata(
    n_vertices: int,
    parcel_labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save the JSON metadata sidecar for vertex projection data."""
    # Summarise per-parcel vertex counts
    parcel_summary = []
    for lbl_int, (lobule, side) in sorted(DEFAULT_SUIT_LABELS.items()):
        count = int((parcel_labels == lbl_int).sum())
        parcel_summary.append({
            "label": lbl_int,
            "name": _build_parcel_label_name(lbl_int),
            "lobule": lobule,
            "hemisphere": side,
            "n_vertices": count,
        })

    metadata = {
        "description": (
            "Per-vertex nuclear projection data for the cerebellar "
            "disconnectome model.  Each surface vertex has a 4-element "
            "probability vector indicating its projection strength to "
            "each deep cerebellar nucleus."
        ),
        "space": "SUIT",
        "n_vertices": n_vertices,
        "n_nuclei": N_NUCLEI,
        "nucleus_order": NUCLEUS_NAMES,
        "depth_sampling": list(DEFAULT_DEPTHS),
        "parcels": parcel_summary,
        "normative_data_source": (
            "Elias et al. (2024) normative structural connectome, "
            "HCP multi-shell diffusion MRI. "
            "DOI: 10.1038/s41597-024-03197-0"
        ),
        "assumptions_reference": "docs/assumptions.md",
        "creation_date": datetime.now(timezone.utc).isoformat(),
        "software_versions": {
            "pipeline": "Cb_Disconnectome v0.2.0",
            "nibabel": nib.__version__,
            "numpy": np.__version__,
        },
    }

    save_metadata(metadata, output_path)
    logger.info("Saved vertex metadata: %s", output_path)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_vertex_projections(
    config: dict | None = None,
) -> Path:
    """
    Build and save per-vertex nuclear projection data.

    For each of the 28,935 SUIT surface vertices:
      1. Samples the cortico-nuclear probability map (Step 1) at multiple
         depths between pial and white surfaces.
      2. Averages across depths to get a (4,) nuclear projection vector.
      3. Determines the SUIT lobular label at the vertex location.

    Also combines the per-nucleus efferent density maps (Step 2) into a
    single 4D NIfTI for use at inference time.

    Parameters
    ----------
    config : dict, optional
        Model configuration overrides.

    Returns
    -------
    Path to the saved vertex_projections.npz file.
    """
    cfg = get_config(config)

    # ------------------------------------------------------------------
    # Step A: Load SUIT surfaces
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step A: Loading SUIT surface meshes")
    logger.info("=" * 60)
    pial_coords, white_coords = load_suit_surfaces(space="SUIT")
    n_vertices = pial_coords.shape[0]

    # ------------------------------------------------------------------
    # Step B: Load the cortico-nuclear probability map (Step 1 output)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step B: Loading cortico-nuclear probability map")
    logger.info("=" * 60)
    cn_map, cn_affine = _load_cortico_nuclear_map()

    # ------------------------------------------------------------------
    # Step C: Sample cortico-nuclear map at each vertex
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step C: Sampling cortico-nuclear map at %d vertices", n_vertices)
    logger.info("=" * 60)
    projections = sample_volume_at_vertices(
        cn_map, cn_affine, pial_coords, white_coords, DEFAULT_DEPTHS,
    )
    # projections shape: (n_vertices, 4)

    # Normalise each vertex's projection vector to sum to 1
    row_sums = projections.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    projections = (projections / row_sums).astype(np.float32)

    n_nonzero = int((projections.sum(axis=1) > 0).sum())
    logger.info(
        "  Vertices with nonzero projections: %d / %d (%.1f%%)",
        n_nonzero, n_vertices, 100.0 * n_nonzero / n_vertices,
    )

    # ------------------------------------------------------------------
    # Step D: Determine per-vertex lobular labels
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step D: Assigning lobular labels to vertices")
    logger.info("=" * 60)
    parc_data, parc_affine, _ = _load_suit_parcellation(cfg)
    parc_data[parc_data > N_CORTICAL_PARCELS] = 0  # zero out nuclei

    # Sample parcellation at vertices using mode-like approach
    # (use depth=0.5, the midpoint of cortical thickness)
    mid_coords = 0.5 * pial_coords + 0.5 * white_coords
    inv_affine = np.linalg.inv(parc_affine)
    ones = np.ones((n_vertices, 1), dtype=np.float64)
    homogeneous = np.hstack([mid_coords, ones])
    voxel_float = (inv_affine @ homogeneous.T).T[:, :3]
    voxel_idx = np.round(voxel_float).astype(int)

    spatial_shape = np.array(parc_data.shape[:3])
    in_bounds = (
        np.all(voxel_idx >= 0, axis=1)
        & (voxel_idx[:, 0] < spatial_shape[0])
        & (voxel_idx[:, 1] < spatial_shape[1])
        & (voxel_idx[:, 2] < spatial_shape[2])
    )

    parcel_labels = np.zeros(n_vertices, dtype=np.int32)
    valid_vox = voxel_idx[in_bounds]
    parcel_labels[in_bounds] = parc_data[
        valid_vox[:, 0], valid_vox[:, 1], valid_vox[:, 2]
    ]

    for lbl_int in sorted(DEFAULT_SUIT_LABELS.keys()):
        count = int((parcel_labels == lbl_int).sum())
        logger.info(
            "  %-20s  %d vertices",
            _build_parcel_label_name(lbl_int), count,
        )

    # ------------------------------------------------------------------
    # Step E: Load and combine efferent density maps
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step E: Loading and combining efferent density maps")
    logger.info("=" * 60)
    efferent_maps, eff_affine = _load_efferent_density_maps()

    # Save combined 4D efferent maps in data/final/ for inference
    efferent_4d_path = DATA_FINAL / "efferent_density_suit_4d.nii.gz"
    save_nifti(efferent_maps, eff_affine, efferent_4d_path)

    # ------------------------------------------------------------------
    # Step F: Save outputs
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step F: Saving outputs")
    logger.info("=" * 60)

    DATA_FINAL.mkdir(parents=True, exist_ok=True)

    # Vertex projections .npz
    npz_path = DATA_FINAL / "vertex_projections.npz"
    np.savez_compressed(
        npz_path,
        projections=projections,
        pial_coords=pial_coords.astype(np.float32),
        white_coords=white_coords.astype(np.float32),
        parcel_labels=parcel_labels,
    )
    logger.info("Saved vertex projections: %s", npz_path)
    logger.info("  projections:   (%d, %d)", *projections.shape)
    logger.info("  pial_coords:   (%d, %d)", *pial_coords.shape)
    logger.info("  white_coords:  (%d, %d)", *white_coords.shape)
    logger.info("  parcel_labels: (%d,)", parcel_labels.shape[0])

    # Metadata sidecar
    meta_path = DATA_FINAL / "vertex_metadata.json"
    save_vertex_metadata(n_vertices, parcel_labels, meta_path)

    logger.info("=" * 60)
    logger.info("Build complete.")
    return npz_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build vertex-level nuclear projections (Step 3)"
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

    build_vertex_projections(config=config)


if __name__ == "__main__":
    main()
