"""
Shared helper functions for the cerebellar disconnectome model.

Provides utilities for:
- File path management
- NIfTI loading with validation
- Coordinate transformations
- Synthetic lesion generation
- Configuration management
"""

import os
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_FINAL = PROJECT_ROOT / "data" / "final"
DOCS_QA = PROJECT_ROOT / "docs" / "qa_reports"

ATLAS_DIR = DATA_RAW / "atlases" / "cerebellar_atlases"
FWT_DIR = DATA_RAW / "atlases" / "FWT"
HCP_DIR = DATA_RAW / "hcp" / "elias2024_connectome"

# ---------------------------------------------------------------------------
# Default model parameters (see docs/assumptions.md A2, A5)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "VERMIS_HALF_WIDTH_MM": 5.0,
    "PARAVERMIS_LATERAL_MM": 15.0,
    "BOUNDARY_TRANSITION_WIDTH_MM": 2.0,
    "EFFERENT_DENSITY_THRESHOLD": 0.01,
    "NUCLEUS_PROB_THRESHOLD": 0.25,
    "SCP_PROB_THRESHOLD": 0.25,
}


def get_config(overrides: dict | None = None) -> dict:
    """Return model configuration, optionally overriding defaults."""
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# NIfTI utilities
# ---------------------------------------------------------------------------

def load_nifti(path: str | Path, dtype: type | None = None) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    Load a NIfTI file and return (data, affine, header).

    Parameters
    ----------
    path : str or Path
        Path to the NIfTI file.
    dtype : type, optional
        Cast the data array to this dtype.

    Returns
    -------
    data : np.ndarray
    affine : np.ndarray of shape (4, 4)
    header : nib.Nifti1Header
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    img = nib.load(str(path))
    data = img.get_fdata()
    if dtype is not None:
        data = data.astype(dtype)
    return data, img.affine, img.header


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str | Path,
               header: nib.Nifti1Header | None = None) -> None:
    """Save a numpy array as a NIfTI file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, str(path))
    logger.info("Saved NIfTI: %s  shape=%s", path, data.shape)


def check_affine_match(affine_a: np.ndarray, affine_b: np.ndarray,
                       atol: float = 1e-4) -> bool:
    """Return True if two affine matrices match within tolerance."""
    return np.allclose(affine_a, affine_b, atol=atol)


def check_shape_match(shape_a: tuple, shape_b: tuple) -> bool:
    """Return True if two image shapes match (ignoring 4th dimension)."""
    return shape_a[:3] == shape_b[:3]


# ---------------------------------------------------------------------------
# Coordinate transformations
# ---------------------------------------------------------------------------

def voxel_to_mm(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel indices to mm coordinates using an affine matrix.

    Parameters
    ----------
    voxel_coords : np.ndarray of shape (..., 3)
        Voxel indices (i, j, k).
    affine : np.ndarray of shape (4, 4)
        Affine matrix mapping voxels to mm.

    Returns
    -------
    mm_coords : np.ndarray of shape (..., 3)
    """
    shape = voxel_coords.shape
    flat = voxel_coords.reshape(-1, 3)
    ones = np.ones((flat.shape[0], 1))
    homogeneous = np.hstack([flat, ones])  # (N, 4)
    mm = (affine @ homogeneous.T).T[:, :3]  # (N, 3)
    return mm.reshape(shape)


def mm_to_voxel(mm_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert mm coordinates to voxel indices using an affine matrix.

    Parameters
    ----------
    mm_coords : np.ndarray of shape (..., 3)
    affine : np.ndarray of shape (4, 4)

    Returns
    -------
    voxel_coords : np.ndarray of shape (..., 3)
    """
    inv_affine = np.linalg.inv(affine)
    return voxel_to_mm(mm_coords, inv_affine)


def get_mm_coordinate_grid(shape: tuple, affine: np.ndarray) -> np.ndarray:
    """
    Build a coordinate grid in mm space for a 3D volume.

    Parameters
    ----------
    shape : tuple of (X, Y, Z)
    affine : np.ndarray of shape (4, 4)

    Returns
    -------
    mm_grid : np.ndarray of shape (X, Y, Z, 3)
        The (x, y, z) mm coordinate of each voxel.
    """
    i, j, k = np.indices(shape)
    coords = np.stack([i, j, k, np.ones_like(i)], axis=-1)  # (X, Y, Z, 4)
    mm_grid = np.einsum("ij,...j->...i", affine, coords)[..., :3]
    return mm_grid


# ---------------------------------------------------------------------------
# Zone classification (see docs/assumptions.md A2)
# ---------------------------------------------------------------------------

def sigmoid_weight(distance_mm: np.ndarray, width: float = 2.0) -> np.ndarray:
    """
    Sigmoid transition function for zone boundary blending.

    Parameters
    ----------
    distance_mm : np.ndarray
        Signed distance from boundary (positive = inside zone).
    width : float
        Transition width in mm.

    Returns
    -------
    weights : np.ndarray
        Values in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-distance_mm / (width / 4.0)))


def classify_zones(x_mm: np.ndarray, config: dict | None = None) -> dict[str, np.ndarray]:
    """
    Classify voxels into vermis / paravermis / lateral hemisphere zones
    based on their x-coordinate (medial-lateral) in SUIT space.

    Returns soft (probabilistic) membership weights for each zone that
    sum to 1.0 at every voxel.

    Parameters
    ----------
    x_mm : np.ndarray
        The x-coordinates in mm (SUIT space, centered near 0).
    config : dict, optional
        Model configuration with zone thresholds.

    Returns
    -------
    dict with keys 'vermis', 'paravermis', 'lateral', each mapping to
    an array of the same shape as x_mm with values in [0, 1].
    """
    cfg = get_config(config)
    vermis_hw = cfg["VERMIS_HALF_WIDTH_MM"]
    para_lat = cfg["PARAVERMIS_LATERAL_MM"]
    trans_w = cfg["BOUNDARY_TRANSITION_WIDTH_MM"]

    abs_x = np.abs(x_mm)

    # Distance from vermis boundary (positive = inside vermis)
    d_vermis = vermis_hw - abs_x
    w_vermis = sigmoid_weight(d_vermis, trans_w)

    # Distance from paravermis outer boundary (positive = inside paravermis)
    d_para_outer = para_lat - abs_x
    w_para_outer = sigmoid_weight(d_para_outer, trans_w)

    # Paravermis = inside para_outer boundary AND outside vermis
    w_paravermis = w_para_outer * (1.0 - w_vermis)

    # Lateral = outside para_outer boundary
    w_lateral = 1.0 - w_para_outer

    # Normalize to sum to 1
    total = w_vermis + w_paravermis + w_lateral
    total = np.where(total > 0, total, 1.0)

    return {
        "vermis": w_vermis / total,
        "paravermis": w_paravermis / total,
        "lateral": w_lateral / total,
    }


# ---------------------------------------------------------------------------
# Synthetic lesion generation
# ---------------------------------------------------------------------------

def make_sphere_lesion(center_mm: tuple | np.ndarray, radius_mm: float,
                       reference_nifti_path: str | Path) -> nib.Nifti1Image:
    """
    Create a binary spherical lesion mask in the space of a reference NIfTI.

    Parameters
    ----------
    center_mm : array-like of shape (3,)
        Center of the sphere in mm coordinates.
    radius_mm : float
        Radius of the sphere in mm.
    reference_nifti_path : str or Path
        Path to a reference NIfTI for affine and shape.

    Returns
    -------
    nib.Nifti1Image
        Binary mask with 1s inside the sphere, 0s outside.
    """
    ref = nib.load(str(reference_nifti_path))
    affine = ref.affine
    shape = ref.shape[:3]

    center_mm = np.asarray(center_mm, dtype=float)
    mm_grid = get_mm_coordinate_grid(shape, affine)

    dist = np.sqrt(((mm_grid - center_mm) ** 2).sum(axis=-1))
    mask = (dist <= radius_mm).astype(np.float32)

    return nib.Nifti1Image(mask, affine, ref.header)


# ---------------------------------------------------------------------------
# Metadata utilities
# ---------------------------------------------------------------------------

def load_metadata(path: str | Path) -> dict:
    """Load JSON metadata sidecar."""
    with open(path) as f:
        return json.load(f)


def save_metadata(data: dict, path: str | Path) -> None:
    """Save JSON metadata sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved metadata: %s", path)


def ensure_directories() -> None:
    """Create all required project directories if they don't exist."""
    for d in [DATA_RAW, DATA_INTERIM, DATA_FINAL, DOCS_QA,
              ATLAS_DIR, FWT_DIR, HCP_DIR,
              DATA_RAW / "suit"]:
        d.mkdir(parents=True, exist_ok=True)
