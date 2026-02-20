"""
Sparse disconnection matrix builder.

Constructs a scipy sparse matrix D of shape (n_suit_voxels, n_vertices) where
D[v, p] represents the probability that lesioning voxel v disconnects surface
vertex p.

The matrix captures four disconnection mechanisms:

  Layer 1 — **Direct cortical injury**: cortical voxels that overlap a vertex's
            cortical location (any depth from pial to white surface).
  Layer 2 — **Nuclear relay disconnection**: nuclear voxels disrupt all vertices
            projecting through that nucleus (weighted by projection probability).
  Layer 3 — **Efferent pathway disruption**: SCP and post-nuclear voxels with
            nonzero efferent density disrupt vertices via the efferent pathway.
  Layer 4 — **Internal WM disconnection**: white matter voxels between cortex
            and nuclei sever cortex-to-nucleus fibers. (Placeholder — requires
            additional tractography or geometric modeling.)

Layers 2 and 3 are combined as a single matrix operation: for any voxel with
efferent density e (shape (8,)), its disconnection contribution to all vertices
is e @ P.T where P is the per-vertex bilateral nuclear projection matrix.

At inference time, for a lesion mask with N nonzero voxels:
    lesion_indices = np.where(lesion_mask.ravel() > 0)[0]
    D_lesion = D[lesion_indices, :]            # sparse row selection
    disruption = D_lesion.max(axis=0).toarray().ravel()  # (n_vertices,)

Output:
    data/final/disconnection_matrix.npz  — sparse CSR matrix components

Usage:
    python -m src.disconnection_matrix [--config config.json]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.utils import (
    DATA_FINAL,
    DATA_INTERIM,
    get_config,
    load_nifti,
    ensure_directories,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Must match the bilateral nucleus ordering in cortico_nuclear_map.py
N_NUCLEI = 8

# Depth fractions for cortical injury mapping (pial=0, white=1)
_CORTICAL_DEPTHS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Minimum efferent density to include in the sparse matrix
_MIN_EFFERENT_THRESHOLD = 1e-4

# Chunk size for processing efferent voxels (controls peak memory)
_EFFERENT_CHUNK_SIZE = 500


# ---------------------------------------------------------------------------
# Layer 1: Direct cortical injury
# ---------------------------------------------------------------------------

def _build_cortical_layer(
    pial_coords: np.ndarray,
    white_coords: np.ndarray,
    suit_affine: np.ndarray,
    suit_shape: tuple[int, int, int],
) -> sp.csr_matrix:
    """
    Build the direct cortical injury layer.

    For each vertex, samples at multiple depths between pial and white
    surfaces to find which voxels overlap the vertex's cortical ribbon.
    If any depth sample falls within voxel v, then D[v, p] = 1.0.

    Returns a sparse CSR matrix of shape (n_total_voxels, n_vertices).
    """
    n_vertices = pial_coords.shape[0]
    n_total_voxels = int(np.prod(suit_shape))
    inv_affine = np.linalg.inv(suit_affine)

    all_rows = []
    all_cols = []

    for depth in _CORTICAL_DEPTHS:
        coords_mm = (1.0 - depth) * pial_coords + depth * white_coords

        # Convert mm to voxel indices
        ones = np.ones((n_vertices, 1), dtype=np.float64)
        homogeneous = np.hstack([coords_mm, ones])
        voxel_float = (inv_affine @ homogeneous.T).T[:, :3]
        voxel_idx = np.round(voxel_float).astype(int)

        # Check bounds
        valid = (
            np.all(voxel_idx >= 0, axis=1)
            & (voxel_idx[:, 0] < suit_shape[0])
            & (voxel_idx[:, 1] < suit_shape[1])
            & (voxel_idx[:, 2] < suit_shape[2])
        )

        if not valid.any():
            continue

        valid_vox = voxel_idx[valid]
        flat_idx = np.ravel_multi_index(
            (valid_vox[:, 0], valid_vox[:, 1], valid_vox[:, 2]),
            suit_shape,
        )
        vertex_indices = np.where(valid)[0]

        all_rows.append(flat_idx)
        all_cols.append(vertex_indices)

    if not all_rows:
        return sp.csr_matrix((n_total_voxels, n_vertices), dtype=np.float32)

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.ones(len(rows), dtype=np.float32)

    D1 = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_total_voxels, n_vertices),
    ).tocsr()

    # Clip duplicates (multiple depths hitting the same voxel) to 1.0
    D1.data = np.minimum(D1.data, 1.0)
    D1.eliminate_zeros()

    n_entries = D1.nnz
    logger.info(
        "Layer 1 (cortical injury): %d nonzero entries, "
        "%d unique voxels affect vertices",
        n_entries, len(np.unique(rows)),
    )
    return D1


# ---------------------------------------------------------------------------
# Layers 2+3: Nuclear + efferent pathway disconnection
# ---------------------------------------------------------------------------

def _build_efferent_layer(
    efferent_data: np.ndarray,
    vertex_projections: np.ndarray,
    suit_shape: tuple[int, int, int],
    threshold: float = _MIN_EFFERENT_THRESHOLD,
) -> sp.csr_matrix:
    """
    Build the nuclear + efferent pathway disconnection layer.

    For each voxel with nonzero efferent density (whether in a nucleus, SCP,
    or downstream), computes the disconnection score for all vertices via:
        score[p] = efferent[v, :] @ vertex_projections[p, :]

    This naturally handles both nuclear voxels (where efferent density
    encodes nuclear identity) and SCP/downstream voxels (where it encodes
    pathway strength).

    Returns a sparse CSR matrix of shape (n_total_voxels, n_vertices).
    """
    n_vertices = vertex_projections.shape[0]
    n_total_voxels = int(np.prod(suit_shape))

    # Flatten efferent to (n_total_voxels, 8)
    E_flat = efferent_data.reshape(-1, N_NUCLEI)

    # Find voxels with any nonzero efferent density
    row_sums = np.abs(E_flat).sum(axis=1)
    nonzero_mask = row_sums > threshold
    nonzero_indices = np.where(nonzero_mask)[0]

    if len(nonzero_indices) == 0:
        logger.warning("No voxels with efferent density above threshold.")
        return sp.csr_matrix((n_total_voxels, n_vertices), dtype=np.float32)

    logger.info(
        "Processing %d voxels with nonzero efferent density "
        "(of %d total, threshold=%.1e)",
        len(nonzero_indices), n_total_voxels, threshold,
    )

    # Transpose projections for efficient matmul: (8, n_vertices)
    P_T = vertex_projections.T.astype(np.float32)

    all_rows = []
    all_cols = []
    all_vals = []

    for chunk_start in range(0, len(nonzero_indices), _EFFERENT_CHUNK_SIZE):
        chunk_end = min(
            chunk_start + _EFFERENT_CHUNK_SIZE, len(nonzero_indices)
        )
        chunk_idx = nonzero_indices[chunk_start:chunk_end]
        E_chunk = E_flat[chunk_idx].astype(np.float32)  # (chunk, 8)

        # scores: (chunk, n_vertices)
        scores = E_chunk @ P_T

        # Sparsify: keep entries above threshold
        r, c = np.where(scores > threshold)
        if len(r) == 0:
            continue

        all_rows.append(chunk_idx[r])
        all_cols.append(c)
        all_vals.append(scores[r, c])

    if not all_rows:
        return sp.csr_matrix((n_total_voxels, n_vertices), dtype=np.float32)

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals).astype(np.float32)

    D23 = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_total_voxels, n_vertices),
    ).tocsr()

    D23.eliminate_zeros()

    logger.info(
        "Layers 2+3 (efferent pathway): %d nonzero entries, "
        "%d unique voxels affect vertices",
        D23.nnz, len(np.unique(rows)),
    )
    return D23


# ---------------------------------------------------------------------------
# Layer 4: Internal WM disconnection (placeholder)
# ---------------------------------------------------------------------------

def _build_wm_layer(
    suit_shape: tuple[int, int, int],
    n_vertices: int,
) -> sp.csr_matrix:
    """
    Build the internal cerebellar white matter disconnection layer.

    This layer captures disconnection of cortex from nuclei via lesions
    in the cerebellar white matter (arbor vitae).  It requires either:
      - Tractography-based modeling: streamlines from cortex to nuclei
      - Geometric modeling: radial projection from cortex through WM to nuclei

    Currently a placeholder that returns an empty sparse matrix.
    A future implementation will populate this layer.

    Returns a sparse CSR matrix of shape (n_total_voxels, n_vertices).
    """
    n_total_voxels = int(np.prod(suit_shape))
    logger.info(
        "Layer 4 (internal WM): placeholder — returning empty matrix. "
        "Future implementation will model cortex-to-nucleus WM disconnection."
    )
    return sp.csr_matrix((n_total_voxels, n_vertices), dtype=np.float32)


# ---------------------------------------------------------------------------
# Matrix combination
# ---------------------------------------------------------------------------

def combine_layers(
    D1: sp.csr_matrix,
    D23: sp.csr_matrix,
    D4: sp.csr_matrix,
) -> sp.csr_matrix:
    """
    Combine disconnection layers into a single sparse matrix.

    Takes the element-wise maximum across layers (a voxel's disconnection
    effect on a vertex is the strongest mechanism).

    All inputs must be CSR matrices of the same shape.
    """
    # For CSR matrices, element-wise max is most easily done via
    # converting to dense for the overlap regions, but since the matrices
    # are very sparse, we can use the maximum operation.
    # scipy doesn't have a direct element-wise max for sparse matrices,
    # so we use the fact that D.maximum(other) works.
    D = D1.maximum(D23).maximum(D4)

    # Clip to [0, 1]
    D = D.tocsr()
    D.data = np.clip(D.data, 0.0, 1.0)

    logger.info(
        "Combined disconnection matrix: shape=%s, nnz=%d, "
        "density=%.4f%%",
        D.shape, D.nnz, 100.0 * D.nnz / (D.shape[0] * D.shape[1]),
    )
    return D


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_disconnection_matrix(
    config: dict | None = None,
) -> sp.csr_matrix:
    """
    Build the full sparse disconnection matrix from precomputed pipeline data.

    Loads outputs from Steps 1-3 and assembles a sparse matrix that maps
    every SUIT-space voxel to the surface vertices it would disconnect.

    Parameters
    ----------
    config : dict, optional
        Model configuration overrides.

    Returns
    -------
    D : scipy.sparse.csr_matrix of shape (n_suit_voxels, n_vertices)
        Disconnection probabilities.
    """
    cfg = get_config(config)

    # ------------------------------------------------------------------
    # Load precomputed data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Building sparse disconnection matrix")
    logger.info("=" * 60)

    # Vertex projections (from Step 3)
    vp_path = DATA_FINAL / "vertex_projections.npz"
    if not vp_path.exists():
        raise FileNotFoundError(
            f"Vertex projections not found: {vp_path}. "
            "Run 'python -m src.build_4d_nifti' first (Step 3)."
        )
    vp = np.load(vp_path)
    vertex_projections = vp["projections"]  # (n_vertices, 8)
    pial_coords = vp["pial_coords"]         # (n_vertices, 3)
    white_coords = vp["white_coords"]       # (n_vertices, 3)
    n_vertices = vertex_projections.shape[0]
    logger.info("Loaded vertex projections: %d vertices", n_vertices)

    # Efferent density maps (from Step 3, in SUIT space)
    eff_path = DATA_FINAL / "efferent_density_suit_4d.nii.gz"
    if not eff_path.exists():
        raise FileNotFoundError(
            f"Efferent density maps not found: {eff_path}. "
            "Run 'python -m src.build_4d_nifti' first (Step 3)."
        )
    eff_data, eff_affine, _ = load_nifti(eff_path, dtype=np.float32)
    suit_shape = eff_data.shape[:3]
    logger.info("Loaded efferent density: shape=%s", eff_data.shape)

    # ------------------------------------------------------------------
    # Build layers
    # ------------------------------------------------------------------

    logger.info("-" * 40)
    logger.info("Building Layer 1: direct cortical injury")
    D1 = _build_cortical_layer(
        pial_coords, white_coords, eff_affine, suit_shape,
    )

    logger.info("-" * 40)
    logger.info("Building Layers 2+3: nuclear + efferent pathway")
    D23 = _build_efferent_layer(
        eff_data, vertex_projections, suit_shape,
    )

    logger.info("-" * 40)
    logger.info("Building Layer 4: internal WM disconnection")
    D4 = _build_wm_layer(suit_shape, n_vertices)

    # ------------------------------------------------------------------
    # Combine layers
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("Combining layers")
    D = combine_layers(D1, D23, D4)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = DATA_FINAL / "disconnection_matrix.npz"
    DATA_FINAL.mkdir(parents=True, exist_ok=True)

    # Save as scipy sparse NPZ
    sp.save_npz(output_path, D)

    # Also save metadata
    meta = {
        "description": (
            "Sparse disconnection matrix mapping SUIT-space voxels to "
            "SUIT surface vertices. D[v, p] is the probability that "
            "lesioning voxel v disconnects vertex p."
        ),
        "suit_shape": list(suit_shape),
        "n_vertices": n_vertices,
        "n_nuclei": N_NUCLEI,
        "nnz": int(D.nnz),
        "density_pct": float(100.0 * D.nnz / (D.shape[0] * D.shape[1])),
        "layers": {
            "1_cortical": {"nnz": int(D1.nnz), "description": "Direct cortical injury"},
            "23_efferent": {"nnz": int(D23.nnz), "description": "Nuclear + efferent pathway"},
            "4_wm": {"nnz": int(D4.nnz), "description": "Internal WM (placeholder)"},
        },
        "matrix_format": "CSR (compressed sparse row)",
        "file_format": "scipy.sparse NPZ",
    }

    meta_path = DATA_FINAL / "disconnection_matrix_metadata.json"
    import json
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
    logger.info("Saved disconnection matrix: %s (%.1f MB)", output_path, size_mb)
    logger.info("Saved metadata: %s", meta_path)
    logger.info("=" * 60)
    logger.info("Disconnection matrix build complete.")

    return D


# ---------------------------------------------------------------------------
# Loading utility
# ---------------------------------------------------------------------------

def load_disconnection_matrix(
    path: str | Path | None = None,
) -> sp.csr_matrix:
    """
    Load a precomputed disconnection matrix.

    Parameters
    ----------
    path : str or Path, optional
        Path to the .npz file. Defaults to data/final/disconnection_matrix.npz.

    Returns
    -------
    D : scipy.sparse.csr_matrix
    """
    if path is None:
        path = DATA_FINAL / "disconnection_matrix.npz"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Disconnection matrix not found: {path}. "
            "Run 'python -m src.disconnection_matrix' to build it."
        )
    D = sp.load_npz(path)
    logger.info(
        "Loaded disconnection matrix: shape=%s, nnz=%d",
        D.shape, D.nnz,
    )
    return D


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build sparse disconnection matrix"
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

    ensure_directories()
    build_disconnection_matrix(config=config)


if __name__ == "__main__":
    main()
