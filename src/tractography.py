"""
Efferent tractography density map builder (Step 2).

Traces and filters streamlines from each deep cerebellar nucleus through the
superior cerebellar peduncle (SCP) and downstream, using the Elias et al.
(2024) normative structural connectome (HCP-derived).

For each nuclear subregion (fastigial, emboliform, globose, dentate), this
module produces a 3D probability map showing where its efferent fibers travel.

Pipeline:
    1. Load whole-brain streamlines (.trk / .tck) from the normative connectome
    2. Load or create nuclear seed masks in MNI space
    3. Filter streamlines that pass through each nuclear ROI
    4. Further filter to keep only efferent streamlines passing through SCP
    5. Convert surviving streamlines to voxel-wise density maps
    6. Normalize each density map to probability (0-1)
    7. Save one density map per nucleus to data/interim/

Output files:
    data/interim/efferent_density_fastigial.nii.gz
    data/interim/efferent_density_emboliform.nii.gz
    data/interim/efferent_density_globose.nii.gz
    data/interim/efferent_density_dentate.nii.gz

Usage:
    python -m src.tractography [--config config.json] [--connectome-dir PATH]
"""

import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from src.utils import (
    ATLAS_DIR,
    DATA_INTERIM,
    FWT_DIR,
    HCP_DIR,
    get_config,
    load_nifti,
    save_nifti,
    ensure_directories,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Deep cerebellar nuclei in processing order
NUCLEUS_NAMES = ["fastigial", "emboliform", "globose", "dentate"]


# ---------------------------------------------------------------------------
# Streamline I/O
# ---------------------------------------------------------------------------

def find_streamline_file(directory: str | Path) -> Path:
    """
    Search for a streamline file (.trk or .tck) in a directory.

    Searches recursively, preferring .trk format (native to DIPY) and
    falling back to .tck (MRtrix format).

    Parameters
    ----------
    directory : str or Path
        Directory to search for streamline files.

    Returns
    -------
    Path
        Path to the first streamline file found.

    Raises
    ------
    FileNotFoundError
        If no streamline file is found.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"Connectome directory does not exist: {directory}"
        )

    # Prefer .trk (DIPY native), then .tck (MRtrix)
    for extension in ["*.trk", "*.tck"]:
        candidates = sorted(directory.rglob(extension))
        if candidates:
            logger.info(
                "Found %d %s file(s) in %s; using: %s",
                len(candidates), extension, directory, candidates[0].name,
            )
            return candidates[0]

    raise FileNotFoundError(
        f"No streamline files (.trk or .tck) found in {directory}. "
        "Download the Elias et al. (2024) connectome first "
        "(python -m src.download --step hcp)."
    )


def load_streamlines(connectome_path: str | Path) -> tuple:
    """
    Load streamline data from a .trk or .tck file.

    Uses DIPY's streamline loading utilities. For .trk files, the header
    contains the reference affine and voxel dimensions. For .tck files,
    an accompanying NIfTI reference may be needed.

    Parameters
    ----------
    connectome_path : str or Path
        Path to a .trk or .tck streamline file.

    Returns
    -------
    streamlines : dipy.io.streamline.ArraySequence
        The loaded streamlines in world (mm) coordinates.
    header : dict
        Header information from the streamline file.
    reference_img : nibabel.Nifti1Image or None
        Reference NIfTI image (available for .trk; None for .tck if no
        companion image is found).
    """
    from dipy.io.streamline import load_trk, load_tck

    connectome_path = Path(connectome_path)
    suffix = connectome_path.suffix.lower()

    if suffix == ".trk":
        logger.info("Loading .trk streamlines: %s", connectome_path)
        trk_file = load_trk(
            str(connectome_path),
            reference="same",
            bbox_valid_check=False,
        )

        # Discard streamlines with coordinates outside the valid bounding box
        # (some normative connectomes have a few streamlines that extend
        # slightly past the volume boundary).
        from dipy.io.streamline import StatefulTractogram
        n_before = len(trk_file.streamlines)
        trk_file.remove_invalid_streamlines()
        n_after = len(trk_file.streamlines)
        if n_before != n_after:
            logger.warning(
                "Removed %d invalid streamlines (out-of-bounds); "
                "%d remaining.", n_before - n_after, n_after,
            )

        streamlines = trk_file.streamlines
        header = dict(trk_file.header)

        # Build a reference NIfTI from the .trk header
        voxel_sizes = trk_file.header["voxel_sizes"]
        dimensions = trk_file.header["dimensions"]
        affine = trk_file.affine
        ref_data = np.zeros(dimensions.astype(int), dtype=np.uint8)
        reference_img = nib.Nifti1Image(ref_data, affine)

        logger.info(
            "Loaded %d streamlines from .trk (voxel_sizes=%s, dims=%s)",
            len(streamlines), voxel_sizes, dimensions,
        )
        return streamlines, header, reference_img

    elif suffix == ".tck":
        logger.info("Loading .tck streamlines: %s", connectome_path)
        tck_file = load_tck(str(connectome_path))
        streamlines = tck_file.streamlines
        header = {}

        # Look for companion NIfTI for reference geometry
        reference_img = None
        parent = connectome_path.parent
        ref_candidates = (
            list(parent.glob("*.nii.gz")) + list(parent.glob("*.nii"))
        )
        if ref_candidates:
            ref_path = ref_candidates[0]
            logger.info("Using companion NIfTI as reference: %s", ref_path)
            reference_img = nib.load(str(ref_path))

        logger.info("Loaded %d streamlines from .tck", len(streamlines))
        return streamlines, header, reference_img

    else:
        raise ValueError(
            f"Unsupported streamline format '{suffix}'. "
            "Expected .trk or .tck."
        )


# ---------------------------------------------------------------------------
# Nuclear masks
# ---------------------------------------------------------------------------

def create_nuclear_masks_mni(
    nuclei_dir: str | Path | None = None,
    mni_reference: nib.Nifti1Image | None = None,
) -> dict[str, nib.Nifti1Image]:
    """
    Create or load nuclear masks in MNI space for each deep cerebellar nucleus.

    Strategy:
        1. Search for pre-existing MNI-space nuclear masks in the atlas
           directory (files matching *space-MNI* and *{nucleus_name}*).
        2. If MNI masks are not found, look for SUIT-space masks and
           resample them to MNI space using nilearn.image.resample_to_img.
        3. If neither is available, extract nuclear regions from the SUIT
           anatomical parcellation (labels for Dentate, Interposed, Fastigial).

    Parameters
    ----------
    nuclei_dir : str or Path, optional
        Directory containing nuclear mask files. Defaults to ATLAS_DIR.
    mni_reference : nibabel.Nifti1Image, optional
        MNI-space reference image defining the target grid. If None, the
        MNI152 2mm template from ATLAS_DIR is used as fallback.

    Returns
    -------
    dict mapping nucleus name -> nibabel.Nifti1Image
        Binary (or probabilistic) mask for each nucleus in MNI space.
    """
    from nilearn.image import resample_to_img

    nuclei_dir = Path(nuclei_dir) if nuclei_dir else ATLAS_DIR

    # Search patterns for each nucleus.
    # "emboliform" and "globose" are the anterior/posterior interposed nuclei;
    # some atlases label them collectively as "interposed".
    search_aliases = {
        "fastigial": ["fastigial", "Fastigial"],
        "emboliform": ["emboliform", "Emboliform", "interposed", "Interposed"],
        "globose": ["globose", "Globose", "interposed", "Interposed"],
        "dentate": ["dentate", "Dentate"],
    }

    masks = {}

    for nucleus in NUCLEUS_NAMES:
        logger.info("Preparing MNI mask for: %s", nucleus)

        # --- Strategy 1: Look for MNI-space mask directly ---
        mni_mask_path = _find_mask_file(
            nuclei_dir, search_aliases[nucleus], space="MNI"
        )
        if mni_mask_path is not None:
            logger.info("  Found MNI mask: %s", mni_mask_path)
            masks[nucleus] = nib.load(str(mni_mask_path))
            continue

        # --- Strategy 2: Resample SUIT-space mask to MNI ---
        suit_mask_path = _find_mask_file(
            nuclei_dir, search_aliases[nucleus], space="SUIT"
        )
        if suit_mask_path is not None:
            logger.info("  Found SUIT mask, resampling to MNI: %s", suit_mask_path)
            suit_img = nib.load(str(suit_mask_path))
            if mni_reference is not None:
                mni_img = resample_to_img(
                    suit_img, mni_reference,
                    interpolation="nearest",
                )
            else:
                mni_ref = _get_mni_reference(nuclei_dir)
                mni_img = resample_to_img(
                    suit_img, mni_ref,
                    interpolation="nearest",
                )
            masks[nucleus] = mni_img
            continue

        # --- Strategy 3: Extract from SUIT anatomical parcellation ---
        logger.info(
            "  No dedicated mask found. Extracting from parcellation..."
        )
        masks[nucleus] = _extract_nucleus_from_parcellation(
            nucleus, nuclei_dir, mni_reference
        )

    return masks


def _find_mask_file(
    directory: Path,
    aliases: list[str],
    space: str,
) -> Path | None:
    """
    Search for a NIfTI mask file matching given aliases and space.

    Parameters
    ----------
    directory : Path
        Root directory to search recursively.
    aliases : list of str
        Name fragments to match (e.g. ['fastigial', 'Fastigial']).
    space : str
        Space label to require in the filename (e.g. 'MNI', 'SUIT').

    Returns
    -------
    Path or None
        Path to the first matching file, or None.
    """
    for alias in aliases:
        pattern = f"*{alias}*space-{space}*.nii*"
        candidates = sorted(directory.rglob(pattern))
        if candidates:
            return candidates[0]

        # Also try with space tag anywhere in the path
        pattern_alt = f"*space-{space}*{alias}*.nii*"
        candidates = sorted(directory.rglob(pattern_alt))
        if candidates:
            return candidates[0]

    return None


def _get_mni_reference(atlas_dir: Path) -> nib.Nifti1Image:
    """
    Load an MNI-space reference image from the atlas directory.

    Searches for any MNI-space NIfTI to use as a spatial reference grid.
    Falls back to creating a standard MNI152 2mm grid.

    Returns
    -------
    nibabel.Nifti1Image
        Reference image in MNI space.
    """
    # Look for any existing MNI-space image
    mni_candidates = sorted(atlas_dir.rglob("*space-MNI*.nii*"))
    if mni_candidates:
        logger.info("Using MNI reference: %s", mni_candidates[0])
        return nib.load(str(mni_candidates[0]))

    # Fallback: build a standard MNI152 2mm grid
    logger.info("No MNI reference found; constructing standard 2mm grid.")
    affine = np.array([
        [-2.,  0.,  0.,  90.],
        [ 0.,  2.,  0., -126.],
        [ 0.,  0.,  2.,  -72.],
        [ 0.,  0.,  0.,    1.],
    ])
    shape = (91, 109, 91)
    data = np.zeros(shape, dtype=np.uint8)
    return nib.Nifti1Image(data, affine)


def _extract_nucleus_from_parcellation(
    nucleus: str,
    atlas_dir: Path,
    mni_reference: nib.Nifti1Image | None,
) -> nib.Nifti1Image:
    """
    Extract a nuclear mask from the SUIT anatomical parcellation.

    The SUIT parcellation labels include entries for Dentate, Interposed,
    and Fastigial nuclei. This function extracts the relevant labels and
    creates a binary mask, resampling to MNI space if needed.

    Parameters
    ----------
    nucleus : str
        One of 'fastigial', 'emboliform', 'globose', 'dentate'.
    atlas_dir : Path
        Directory containing the SUIT parcellation.
    mni_reference : nibabel.Nifti1Image or None
        MNI-space reference for resampling.

    Returns
    -------
    nibabel.Nifti1Image
        Binary mask in MNI space.
    """
    from nilearn.image import resample_to_img

    # Label indices from the standard SUIT anatomical parcellation.
    # Left and Right labels for each nucleus.
    nucleus_labels = {
        "fastigial": [33, 34],       # Left/Right Fastigial
        "dentate": [29, 30],          # Left/Right Dentate
        "emboliform": [31, 32],       # Left/Right Interposed (anterior part)
        "globose": [31, 32],          # Left/Right Interposed (posterior part)
    }

    # Find the parcellation file (prefer MNI, then SUIT)
    parc_path = None
    for space in ["MNI", "SUIT"]:
        candidates = sorted(atlas_dir.rglob(f"*Anatom*space-{space}*dseg.nii*"))
        if candidates:
            parc_path = candidates[0]
            break

    if parc_path is None:
        raise FileNotFoundError(
            f"SUIT anatomical parcellation not found in {atlas_dir}. "
            "Run src/download.py first."
        )

    logger.info("  Extracting '%s' from parcellation: %s", nucleus, parc_path)
    parc_img = nib.load(str(parc_path))
    parc_data = np.asarray(parc_img.dataobj)

    labels = nucleus_labels[nucleus]
    mask_data = np.isin(parc_data, labels).astype(np.float32)

    # For emboliform/globose (both map to "interposed"), split the mask
    # along the anterior-posterior axis. Emboliform is anterior, globose
    # is posterior within the interposed nucleus.
    if nucleus in ("emboliform", "globose"):
        mask_voxels = np.argwhere(mask_data > 0)
        if mask_voxels.size > 0:
            # Determine anterior-posterior axis (typically the y-axis
            # in MNI/SUIT coordinates)
            y_coords = mask_voxels[:, 1]
            y_median = np.median(y_coords)

            if nucleus == "emboliform":
                # Anterior interposed: voxels with y >= median
                # (more anterior in standard orientation)
                ap_mask = np.zeros_like(mask_data)
                ap_mask[mask_voxels[y_coords >= y_median, 0],
                        mask_voxels[y_coords >= y_median, 1],
                        mask_voxels[y_coords >= y_median, 2]] = 1.0
                mask_data = ap_mask
            else:
                # Posterior interposed (globose): y < median
                ap_mask = np.zeros_like(mask_data)
                ap_mask[mask_voxels[y_coords < y_median, 0],
                        mask_voxels[y_coords < y_median, 1],
                        mask_voxels[y_coords < y_median, 2]] = 1.0
                mask_data = ap_mask

    mask_img = nib.Nifti1Image(mask_data, parc_img.affine, parc_img.header)

    # Resample to MNI if the parcellation is in SUIT space
    if "SUIT" in str(parc_path):
        ref = mni_reference if mni_reference else _get_mni_reference(atlas_dir)
        mask_img = resample_to_img(mask_img, ref, interpolation="nearest")

    n_voxels = int((np.asarray(mask_img.dataobj) > 0).sum())
    logger.info("  '%s' mask: %d nonzero voxels", nucleus, n_voxels)

    if n_voxels == 0:
        logger.warning(
            "  Nuclear mask for '%s' is empty. Check atlas labels.", nucleus
        )

    return mask_img


# ---------------------------------------------------------------------------
# SCP mask
# ---------------------------------------------------------------------------

def _load_scp_mask(fwt_dir: Path | None = None) -> nib.Nifti1Image:
    """
    Load the superior cerebellar peduncle (SCP) mask from the FWT atlas.

    The FWT atlas (Radwan et al., 2022) provides probability maps for
    major white matter tracts in MNI space. We search for files matching
    *SCP* in the FWT directory.

    Parameters
    ----------
    fwt_dir : Path, optional
        Directory containing FWT atlas data. Defaults to FWT_DIR.

    Returns
    -------
    nibabel.Nifti1Image
        SCP probability or binary mask in MNI space.

    Raises
    ------
    FileNotFoundError
        If no SCP mask file is found.
    """
    fwt_dir = Path(fwt_dir) if fwt_dir else FWT_DIR

    if not fwt_dir.exists():
        raise FileNotFoundError(
            f"FWT atlas directory does not exist: {fwt_dir}. "
            "Run src/download.py --step fwt first."
        )

    # Search for SCP files (may be named SCP, scp, or
    # superior_cerebellar_peduncle)
    scp_candidates = sorted(fwt_dir.rglob("*SCP*.nii*"))
    if not scp_candidates:
        scp_candidates = sorted(fwt_dir.rglob("*scp*.nii*"))
    if not scp_candidates:
        scp_candidates = sorted(
            fwt_dir.rglob("*superior_cerebellar_peduncle*.nii*")
        )

    if not scp_candidates:
        raise FileNotFoundError(
            f"No SCP mask found in {fwt_dir}. "
            "Expected a file matching *SCP*.nii* in the FWT atlas."
        )

    scp_path = scp_candidates[0]
    logger.info("Loading SCP mask: %s", scp_path)
    scp_img = nib.load(str(scp_path))

    n_nonzero = int((np.asarray(scp_img.dataobj) > 0).sum())
    logger.info("SCP mask: %d nonzero voxels", n_nonzero)

    return scp_img


# ---------------------------------------------------------------------------
# Streamline filtering
# ---------------------------------------------------------------------------

def filter_efferent_streamlines(
    streamlines,
    nucleus_mask_img: nib.Nifti1Image,
    scp_mask_img: nib.Nifti1Image,
    config: dict | None = None,
):
    """
    Filter streamlines to keep only efferent fibers from a given nucleus.

    A streamline is considered efferent from a nucleus if it:
        1. Passes through the nuclear ROI mask (seed criterion)
        2. Also passes through the SCP mask (pathway criterion)

    Both masks may be probabilistic; thresholds from config determine the
    binarization cutoff.

    Parameters
    ----------
    streamlines : dipy ArraySequence or list of arrays
        Input streamlines in world (mm) coordinates.
    nucleus_mask_img : nibabel.Nifti1Image
        Nuclear region mask in MNI space.
    scp_mask_img : nibabel.Nifti1Image
        SCP mask in MNI space.
    config : dict, optional
        Model configuration. Uses keys:
        - NUCLEUS_PROB_THRESHOLD : float (default 0.25)
        - SCP_PROB_THRESHOLD : float (default 0.25)

    Returns
    -------
    filtered_streamlines : list of np.ndarray
        Streamlines passing through both the nucleus and SCP.
    n_nucleus : int
        Number of streamlines passing through the nucleus (before SCP filter).
    n_efferent : int
        Number of streamlines passing through both nucleus and SCP.
    """
    from dipy.tracking.utils import target
    from nilearn.image import resample_to_img

    cfg = get_config(config)
    nucleus_threshold = cfg["NUCLEUS_PROB_THRESHOLD"]
    scp_threshold = cfg["SCP_PROB_THRESHOLD"]

    # --- Prepare nucleus mask ---
    nucleus_data = np.asarray(nucleus_mask_img.dataobj)
    nucleus_affine = nucleus_mask_img.affine
    nucleus_binary = (nucleus_data >= nucleus_threshold).astype(np.uint8)

    # --- Prepare SCP mask ---
    # Ensure SCP mask is in the same space/grid as the nucleus mask
    scp_data = np.asarray(scp_mask_img.dataobj)
    scp_affine = scp_mask_img.affine

    if scp_data.shape[:3] != nucleus_data.shape[:3] or not np.allclose(
        scp_affine, nucleus_affine, atol=1e-3
    ):
        logger.info("  Resampling SCP mask to match nucleus mask grid...")
        scp_resampled = resample_to_img(
            scp_mask_img, nucleus_mask_img, interpolation="nearest"
        )
        scp_data = np.asarray(scp_resampled.dataobj)

    scp_binary = (scp_data >= scp_threshold).astype(np.uint8)

    # --- Step 1: Filter streamlines through nucleus ---
    logger.info(
        "  Filtering through nucleus mask "
        "(threshold=%.2f, %d nonzero voxels)...",
        nucleus_threshold, int(nucleus_binary.sum()),
    )
    nucleus_filtered = list(
        target(streamlines, nucleus_affine, nucleus_binary, include=True)
    )
    n_nucleus = len(nucleus_filtered)
    logger.info("  Streamlines through nucleus: %d", n_nucleus)

    if n_nucleus == 0:
        logger.warning("  No streamlines pass through the nucleus mask.")
        return [], 0, 0

    # --- Step 2: Filter those streamlines through SCP ---
    logger.info(
        "  Filtering through SCP mask "
        "(threshold=%.2f, %d nonzero voxels)...",
        scp_threshold, int(scp_binary.sum()),
    )
    efferent_filtered = list(
        target(nucleus_filtered, nucleus_affine, scp_binary, include=True)
    )
    n_efferent = len(efferent_filtered)
    logger.info("  Efferent streamlines (nucleus + SCP): %d", n_efferent)

    return efferent_filtered, n_nucleus, n_efferent


# ---------------------------------------------------------------------------
# Density map computation
# ---------------------------------------------------------------------------

def compute_density_map(
    streamlines,
    reference_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """
    Compute a voxel-wise streamline visitation density map and normalize
    it to a probability map (values in [0, 1]).

    Parameters
    ----------
    streamlines : list of np.ndarray or dipy ArraySequence
        Streamlines in world (mm) coordinates.
    reference_img : nibabel.Nifti1Image
        Reference image defining the output grid (shape and affine).

    Returns
    -------
    nibabel.Nifti1Image
        Normalized density map where 1.0 = maximum visitation.
    """
    from dipy.tracking.utils import density_map

    affine = reference_img.affine
    shape = reference_img.shape[:3]

    if len(streamlines) == 0:
        logger.warning("No streamlines provided; returning empty density map.")
        return nib.Nifti1Image(
            np.zeros(shape, dtype=np.float32), affine
        )

    logger.info(
        "Computing density map from %d streamlines (grid %s)...",
        len(streamlines), shape,
    )

    # DIPY density_map counts how many streamlines visit each voxel
    dm = density_map(streamlines, affine, shape)
    dm = dm.astype(np.float32)

    # Normalize to [0, 1] probability
    max_val = dm.max()
    if max_val > 0:
        dm = dm / max_val

    n_nonzero = int((dm > 0).sum())
    logger.info(
        "Density map: %d nonzero voxels, max raw count before norm = %.0f",
        n_nonzero, max_val,
    )

    return nib.Nifti1Image(dm, affine)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def build_efferent_density_maps(config: dict | None = None) -> dict[str, Path]:
    """
    Build efferent density maps for all deep cerebellar nuclei.

    This is the main orchestration function that:
        1. Locates and loads the normative connectome streamlines
        2. Loads/creates nuclear masks in MNI space
        3. Loads the SCP mask from the FWT atlas
        4. For each nucleus, filters streamlines and builds a density map
        5. Saves results to data/interim/

    Parameters
    ----------
    config : dict, optional
        Model configuration overrides (see src.utils.DEFAULT_CONFIG).

    Returns
    -------
    dict mapping nucleus name -> Path to saved density map NIfTI.
    """
    cfg = get_config(config)
    ensure_directories()

    # --- 1. Load streamlines ---
    logger.info("=" * 60)
    logger.info("STEP 2: Building efferent density maps")
    logger.info("=" * 60)

    streamline_path = find_streamline_file(HCP_DIR)
    streamlines, header, reference_img = load_streamlines(streamline_path)
    total_streamlines = len(streamlines)
    logger.info("Total streamlines in connectome: %d", total_streamlines)

    # --- 2. Load nuclear masks ---
    logger.info("-" * 40)
    logger.info("Loading nuclear masks...")
    nuclear_masks = create_nuclear_masks_mni(
        nuclei_dir=ATLAS_DIR,
        mni_reference=reference_img,
    )

    # --- 3. Load SCP mask ---
    logger.info("-" * 40)
    logger.info("Loading SCP mask from FWT atlas...")
    scp_mask_img = _load_scp_mask(FWT_DIR)

    # --- 4. Process each nucleus ---
    output_paths = {}

    for nucleus in tqdm(NUCLEUS_NAMES, desc="Processing nuclei"):
        logger.info("-" * 40)
        logger.info("Processing nucleus: %s", nucleus)

        nucleus_mask_img = nuclear_masks[nucleus]

        # Filter streamlines
        efferent_sl, n_nuc, n_eff = filter_efferent_streamlines(
            streamlines, nucleus_mask_img, scp_mask_img, cfg,
        )

        # Compute density map
        if reference_img is not None:
            ref = reference_img
        else:
            ref = nucleus_mask_img

        density_img = compute_density_map(efferent_sl, ref)

        # Apply density threshold from config
        density_threshold = cfg["EFFERENT_DENSITY_THRESHOLD"]
        density_data = np.asarray(density_img.dataobj).copy()
        density_data[density_data < density_threshold] = 0.0
        density_img = nib.Nifti1Image(density_data, density_img.affine)

        # Save
        output_path = DATA_INTERIM / f"efferent_density_{nucleus}.nii.gz"
        save_nifti(density_data, density_img.affine, output_path)
        output_paths[nucleus] = output_path

        logger.info(
            "  %s: %d/%d streamlines through nucleus, "
            "%d efferent (through SCP), "
            "%d nonzero density voxels -> %s",
            nucleus, n_nuc, total_streamlines, n_eff,
            int((density_data > 0).sum()),
            output_path.name,
        )

    # --- 5. Summary ---
    logger.info("=" * 60)
    logger.info("Efferent density maps complete. Outputs:")
    for nucleus, path in output_paths.items():
        logger.info("  %s: %s", nucleus, path)
    logger.info("=" * 60)

    return output_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for tractography density map building."""
    parser = argparse.ArgumentParser(
        description="Build efferent density maps from normative tractography (Step 2)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file with parameter overrides",
    )
    parser.add_argument(
        "--connectome-dir", type=str, default=None,
        help=(
            "Path to directory containing connectome streamlines "
            "(default: data/raw/hcp/elias2024_connectome)"
        ),
    )
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override HCP_DIR if user provides a custom path
    if args.connectome_dir:
        import src.utils
        src.utils.HCP_DIR = Path(args.connectome_dir)

    build_efferent_density_maps(config=config)


if __name__ == "__main__":
    main()
