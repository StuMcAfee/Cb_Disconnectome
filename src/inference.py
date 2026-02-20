"""
Lesion inference engine for the cerebellar disconnectome model.

Takes a binary lesion mask in SUIT space and produces **vertex-wise** cortical
disruption probabilities (one per SUIT surface vertex, 28,935 total) by
combining precomputed per-vertex nuclear projections with efferent pathway
density maps.

Input:  Binary lesion mask in SUIT space (NIfTI, 3D)
Output: 1D array of disruption probabilities, one per surface vertex (28,935)

Architecture
------------
Precomputed data (from build_4d_nifti / Step 3):
  - vertex_projections.npz: per-vertex bilateral nuclear projection
    probabilities (N_vertices, 8) plus pial/white surface coordinates
  - efferent_density_suit_4d.nii.gz: combined efferent density maps (X,Y,Z,8)

Bilateral nuclei (8 total, 4 per hemisphere):
  [0-3] left:  fastigial, emboliform, globose, dentate
  [4-7] right: fastigial, emboliform, globose, dentate

At inference time, for a lesion with N voxels:
  1. Extract efferent density at lesion voxels -> E (N_lesion, 8)
  2. Compute per-voxel per-vertex scores -> E @ P.T (N_lesion, N_vertices)
  3. Aggregate across lesion voxels (max/mean/etc.) -> (N_vertices,)
  4. Direct injury: if lesion overlaps a vertex's cortical location, set to 1.0

Aggregation methods:
  - max (default):     maximum weighted pathway density across lesion voxels
  - mean:              average weighted pathway density across lesion voxels
  - weighted_sum:      sum of densities, normalized to [0, 1]
  - threshold_fraction: per-nucleus pathway fraction, combined by projection

Usage:
    python -m src.inference <lesion_SUIT.nii.gz> [output_dir]
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    load_nifti,
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

DEFAULT_EFFERENT_PATH = DATA_FINAL / "efferent_density_suit_4d.nii.gz"
DEFAULT_PROJECTIONS_PATH = DATA_FINAL / "vertex_projections.npz"
DEFAULT_METADATA_PATH = DATA_FINAL / "vertex_metadata.json"

# Threshold for the 'threshold_fraction' method
_THRESHOLD_FRACTION_CUTOFF = 0.1

# Available aggregation methods
VALID_METHODS = ("max", "mean", "weighted_sum", "threshold_fraction")

# Depth fractions for direct injury checking (pial=0 to white=1)
_DIRECT_INJURY_DEPTHS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

N_NUCLEI = 8


# ---------------------------------------------------------------------------
# Vertex projection loader
# ---------------------------------------------------------------------------

def load_vertex_projections(
    projections_path: str | Path = DEFAULT_PROJECTIONS_PATH,
) -> dict[str, np.ndarray]:
    """
    Load precomputed vertex projection data.

    Returns a dict with keys: 'projections', 'pial_coords', 'white_coords',
    'parcel_labels'.
    """
    projections_path = Path(projections_path)
    if not projections_path.exists():
        raise FileNotFoundError(
            f"Vertex projections not found: {projections_path}. "
            "Run 'python -m src.build_4d_nifti' first (Step 3)."
        )
    npz = np.load(projections_path)
    return {
        "projections": npz["projections"],
        "pial_coords": npz["pial_coords"],
        "white_coords": npz["white_coords"],
        "parcel_labels": npz["parcel_labels"],
    }


# ---------------------------------------------------------------------------
# Direct injury detection
# ---------------------------------------------------------------------------

def _detect_direct_injury(
    lesion_data: np.ndarray,
    lesion_affine: np.ndarray,
    pial_coords: np.ndarray,
    white_coords: np.ndarray,
) -> np.ndarray:
    """
    Identify vertices whose cortical locations overlap the lesion.

    For each vertex, samples at multiple depths between the pial and white
    surfaces.  If any depth sample falls inside a lesion voxel, the vertex
    is marked as directly injured.

    Returns a boolean array of shape (N_vertices,).
    """
    n_vertices = pial_coords.shape[0]
    spatial_shape = np.array(lesion_data.shape[:3])
    inv_affine = np.linalg.inv(lesion_affine)

    injured = np.zeros(n_vertices, dtype=bool)

    for depth in _DIRECT_INJURY_DEPTHS:
        coords_mm = (1.0 - depth) * pial_coords + depth * white_coords

        ones = np.ones((n_vertices, 1), dtype=np.float64)
        homogeneous = np.hstack([coords_mm, ones])
        voxel_float = (inv_affine @ homogeneous.T).T[:, :3]
        voxel_idx = np.round(voxel_float).astype(int)

        in_bounds = (
            np.all(voxel_idx >= 0, axis=1)
            & (voxel_idx[:, 0] < spatial_shape[0])
            & (voxel_idx[:, 1] < spatial_shape[1])
            & (voxel_idx[:, 2] < spatial_shape[2])
        )

        if not in_bounds.any():
            continue

        valid_voxels = voxel_idx[in_bounds]
        hits = lesion_data[
            valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]
        ] > 0

        # Map hits back to the full vertex array
        in_bounds_indices = np.where(in_bounds)[0]
        injured[in_bounds_indices[hits]] = True

    return injured


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def infer_disruption(
    lesion_path: str | Path,
    efferent_path: str | Path = DEFAULT_EFFERENT_PATH,
    projections_path: str | Path = DEFAULT_PROJECTIONS_PATH,
    method: str = "max",
) -> np.ndarray:
    """
    Compute vertex-wise cortical disruption from a cerebellar lesion mask.

    Parameters
    ----------
    lesion_path : str or Path
        Path to a binary lesion mask NIfTI in SUIT space (3D).
    efferent_path : str or Path
        Path to the 4D efferent density maps (X, Y, Z, 4).
    projections_path : str or Path
        Path to vertex_projections.npz with per-vertex nuclear projection
        probabilities and surface coordinates.
    method : str
        Aggregation method: 'max', 'mean', 'weighted_sum', or
        'threshold_fraction'.

    Returns
    -------
    disruption : np.ndarray of shape (N_vertices,)
        Disruption probability for each surface vertex.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Choose from: {VALID_METHODS}"
        )

    # --- Load inputs ---
    logger.info("Loading lesion mask: %s", lesion_path)
    lesion_data, lesion_affine, _ = load_nifti(lesion_path, dtype=np.float32)

    logger.info("Loading efferent density maps: %s", efferent_path)
    eff_data, eff_affine, _ = load_nifti(efferent_path, dtype=np.float32)

    logger.info("Loading vertex projections: %s", projections_path)
    vp = load_vertex_projections(projections_path)
    proj = vp["projections"]            # (N_vertices, 4)
    pial_coords = vp["pial_coords"]     # (N_vertices, 3)
    white_coords = vp["white_coords"]   # (N_vertices, 3)

    n_vertices = proj.shape[0]

    # --- Verify spatial alignment ---
    if not check_affine_match(lesion_affine, eff_affine):
        raise ValueError(
            "Affine mismatch between lesion mask and efferent density maps. "
            "Both must be in the same SUIT space."
        )

    if not check_shape_match(lesion_data.shape, eff_data.shape):
        raise ValueError(
            "Shape mismatch between lesion mask and efferent density maps. "
            f"Lesion shape: {lesion_data.shape[:3]}, "
            f"Efferent shape: {eff_data.shape[:3]}"
        )

    # --- Identify lesion voxels ---
    lesion_mask = lesion_data > 0
    n_lesion_voxels = int(lesion_mask.sum())
    logger.info("Lesion voxels: %d", n_lesion_voxels)

    if n_lesion_voxels == 0:
        logger.warning("Lesion mask is empty; returning zero disruption.")
        return np.zeros(n_vertices, dtype=np.float32)

    # --- Extract efferent density at lesion voxels ---
    # E shape: (N_lesion, 4)
    E_lesion = eff_data[lesion_mask]

    # --- Aggregate ---
    if method == "max":
        # (N_lesion, 4) @ (4, N_vertices) -> (N_lesion, N_vertices)
        # then max across lesion voxels
        scores = E_lesion @ proj.T
        disruption = scores.max(axis=0)

    elif method == "mean":
        scores = E_lesion @ proj.T
        disruption = scores.mean(axis=0)

    elif method == "weighted_sum":
        scores = E_lesion @ proj.T
        raw_sum = scores.sum(axis=0)
        max_val = raw_sum.max()
        if max_val > 0:
            disruption = raw_sum / max_val
        else:
            disruption = raw_sum

    elif method == "threshold_fraction":
        # Per-nucleus: fraction of pathway volume intersected by the lesion
        fractions = np.zeros(N_NUCLEI, dtype=np.float64)
        for n in range(N_NUCLEI):
            above = eff_data[..., n] > _THRESHOLD_FRACTION_CUTOFF
            pathway_vol = float(above.sum())
            lesion_intersect = float(above[lesion_mask].sum())
            fractions[n] = lesion_intersect / max(pathway_vol, 1.0)
        # Combine per-nucleus fractions using vertex projection weights
        disruption = (proj @ fractions).astype(np.float32)

    # --- Direct injury ---
    injured = _detect_direct_injury(
        lesion_data, lesion_affine, pial_coords, white_coords,
    )
    n_injured = int(injured.sum())
    if n_injured > 0:
        logger.info("Direct cortical injury: %d vertices", n_injured)
        disruption[injured] = 1.0

    disruption = np.clip(disruption, 0.0, 1.0).astype(np.float32)

    logger.info(
        "Disruption (%s): min=%.4f, max=%.4f, mean=%.4f, "
        "nonzero=%d / %d vertices",
        method, disruption.min(), disruption.max(), disruption.mean(),
        int((disruption > 0).sum()), n_vertices,
    )

    return disruption


# ---------------------------------------------------------------------------
# Sparse matrix inference
# ---------------------------------------------------------------------------

DEFAULT_MATRIX_PATH = DATA_FINAL / "disconnection_matrix.npz"


def infer_disruption_sparse(
    lesion_path: str | Path,
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
    efferent_path: str | Path = DEFAULT_EFFERENT_PATH,
    projections_path: str | Path = DEFAULT_PROJECTIONS_PATH,
    method: str = "max",
) -> np.ndarray:
    """
    Compute vertex-wise cortical disruption using the sparse disconnection
    matrix.  This method captures ALL disconnection mechanisms including
    cortical injury, nuclear relay, efferent pathway, and (when available)
    internal cerebellar white matter disconnection.

    Parameters
    ----------
    lesion_path : str or Path
        Path to a binary lesion mask NIfTI in SUIT space (3D).
    matrix_path : str or Path
        Path to the sparse disconnection matrix (.npz).
    efferent_path : str or Path
        Path to the 4D efferent density maps (for shape/affine reference).
    projections_path : str or Path
        Path to vertex_projections.npz (for n_vertices reference).
    method : str
        Aggregation method: 'max', 'mean', or 'weighted_sum'.
        'threshold_fraction' is not supported with sparse matrix inference.

    Returns
    -------
    disruption : np.ndarray of shape (N_vertices,)
        Disruption probability for each surface vertex.
    """
    import scipy.sparse as sp

    valid_sparse_methods = ("max", "mean", "weighted_sum")
    if method not in valid_sparse_methods:
        raise ValueError(
            f"Sparse matrix inference does not support method '{method}'. "
            f"Choose from: {valid_sparse_methods}"
        )

    # --- Load inputs ---
    logger.info("Loading lesion mask: %s", lesion_path)
    lesion_data, lesion_affine, _ = load_nifti(lesion_path, dtype=np.float32)

    logger.info("Loading disconnection matrix: %s", matrix_path)
    D = sp.load_npz(str(matrix_path))
    n_total_voxels, n_vertices = D.shape

    # Verify the lesion grid matches the matrix
    lesion_n_voxels = int(np.prod(lesion_data.shape[:3]))
    if lesion_n_voxels != n_total_voxels:
        raise ValueError(
            f"Lesion voxel count ({lesion_n_voxels}) does not match "
            f"disconnection matrix rows ({n_total_voxels}). "
            "Both must be in the same SUIT space grid."
        )

    # --- Identify lesion voxels ---
    lesion_flat = lesion_data.ravel() > 0
    lesion_indices = np.where(lesion_flat)[0]
    n_lesion = len(lesion_indices)
    logger.info("Lesion voxels: %d", n_lesion)

    if n_lesion == 0:
        logger.warning("Lesion mask is empty; returning zero disruption.")
        return np.zeros(n_vertices, dtype=np.float32)

    # --- Extract lesion rows from the sparse matrix ---
    D_lesion = D[lesion_indices, :]  # (n_lesion, n_vertices) sparse

    # --- Aggregate ---
    if method == "max":
        # Element-wise max across lesion voxels
        # .max(axis=0) on sparse returns a dense matrix; convert to array
        disruption = np.asarray(D_lesion.max(axis=0).todense()).ravel()
    elif method == "mean":
        disruption = np.asarray(D_lesion.mean(axis=0)).ravel()
    elif method == "weighted_sum":
        raw_sum = np.asarray(D_lesion.sum(axis=0)).ravel()
        max_val = float(raw_sum.max())
        if max_val > 0:
            disruption = raw_sum / max_val
        else:
            disruption = raw_sum

    # Ensure dense float array before clipping
    disruption = np.asarray(disruption, dtype=np.float64).ravel()
    disruption = np.clip(disruption, 0.0, 1.0).astype(np.float32)

    logger.info(
        "Disruption (sparse, %s): min=%.4f, max=%.4f, mean=%.4f, "
        "nonzero=%d / %d vertices",
        method, disruption.min(), disruption.max(), disruption.mean(),
        int((disruption > 0).sum()), n_vertices,
    )

    return disruption


# ---------------------------------------------------------------------------
# Summary table (aggregated by lobule)
# ---------------------------------------------------------------------------

def summarize_disruption(
    disruption: np.ndarray,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    projections_path: str | Path = DEFAULT_PROJECTIONS_PATH,
) -> pd.DataFrame:
    """
    Aggregate vertex-level disruption by lobule and return a sorted DataFrame.

    For each of the 28 SUIT lobular parcels, computes the mean disruption
    across its vertices.

    Parameters
    ----------
    disruption : np.ndarray of shape (N_vertices,)
        Disruption probability per vertex.
    metadata_path : str or Path
        Path to vertex_metadata.json.
    projections_path : str or Path
        Path to vertex_projections.npz (for parcel_labels).

    Returns
    -------
    pd.DataFrame
        Columns: parcel_name, disruption_mean, disruption_max, n_vertices.
        Sorted by disruption_mean in descending order.
    """
    meta = load_metadata(metadata_path)
    vp = load_vertex_projections(projections_path)
    parcel_labels = vp["parcel_labels"]

    rows = []
    for parcel_info in meta.get("parcels", []):
        lbl = parcel_info["label"]
        name = parcel_info["name"]
        mask = parcel_labels == lbl
        n_verts = int(mask.sum())

        if n_verts > 0:
            d_mean = float(disruption[mask].mean())
            d_max = float(disruption[mask].max())
        else:
            d_mean = 0.0
            d_max = 0.0

        rows.append({
            "parcel_name": name,
            "disruption_mean": d_mean,
            "disruption_max": d_max,
            "n_vertices": n_verts,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("disruption_mean", ascending=False).reset_index(drop=True)

    if len(df) > 0:
        logger.info(
            "Summary: %d parcels, top disruption = %.4f (%s)",
            len(df), df["disruption_mean"].iloc[0], df["parcel_name"].iloc[0],
        )

    return df


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_inference(
    lesion_path: str | Path,
    output_dir: str | Path | None = None,
    methods: list[str] | None = None,
    use_sparse: bool = False,
) -> dict[str, np.ndarray]:
    """
    Run lesion inference with one or more aggregation methods.

    Parameters
    ----------
    lesion_path : str or Path
        Path to a binary lesion mask NIfTI in SUIT space.
    output_dir : str or Path, optional
        Directory for saving outputs.
    methods : list of str, optional
        Aggregation methods to run. Defaults to all available methods.
    use_sparse : bool
        If True, use the sparse disconnection matrix for inference
        (captures all disconnection mechanisms including internal WM).
        If False (default), use the legacy matmul approach.

    Returns
    -------
    dict mapping method name to disruption array (N_vertices,).
    """
    lesion_path = Path(lesion_path)
    if methods is None:
        methods = list(VALID_METHODS)

    for m in methods:
        if m not in VALID_METHODS:
            raise ValueError(f"Unknown method '{m}'. Choose from: {VALID_METHODS}")

    logger.info("=" * 60)
    logger.info("Cerebellar Disconnectome Inference (vertex-level)")
    logger.info("=" * 60)
    logger.info("Lesion mask:   %s", lesion_path)
    logger.info("Methods:       %s", methods)
    logger.info("Backend:       %s", "sparse matrix" if use_sparse else "matmul")

    results = {}

    for method in methods:
        logger.info("-" * 40)
        logger.info("Running method: %s", method)

        if use_sparse and method != "threshold_fraction":
            disruption = infer_disruption_sparse(lesion_path, method=method)
        else:
            disruption = infer_disruption(lesion_path, method=method)
        results[method] = disruption

    # --- Save outputs if requested ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        lesion_stem = lesion_path.stem.replace(".nii", "")

        # Save vertex-level disruption arrays (.npz)
        arrays_dict = {f"disruption_{m}": arr for m, arr in results.items()}
        npz_path = output_dir / f"{lesion_stem}_vertex_disruption.npz"
        np.savez_compressed(npz_path, **arrays_dict)
        logger.info("Saved vertex disruption: %s", npz_path)

        # Save lobule-level summary CSV
        try:
            for method, disruption in results.items():
                df = summarize_disruption(disruption)
                csv_path = output_dir / f"{lesion_stem}_summary_{method}.csv"
                df.to_csv(csv_path, index=False, float_format="%.6f")
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
