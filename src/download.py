"""
Data acquisition scripts for the cerebellar disconnectome model.

Downloads all required atlas files and normative tractography data:
1. SUIT cerebellar atlas collection (Diedrichsen lab)
2. SUIT template and flatmap files (via SUITPy)
3. Radwan et al. (2022) Fun With Tracts (FWT) white matter atlas
4. Elias et al. (2024) normative structural connectome (HCP-derived)
5. Deep cerebellar nuclei atlas verification (dentate, emboliform, globose, fastigial)

Usage:
    python -m src.download [--step {all,atlases,suit,fwt,hcp,nuclei,checksums,verify}]
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

from src.utils import (
    ATLAS_DIR,
    DATA_RAW,
    FWT_DIR,
    HCP_DIR,
    PROJECT_ROOT,
    ensure_directories,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Repository URLs and remote data references
# ---------------------------------------------------------------------------

CEREBELLAR_ATLASES_REPO = (
    "https://github.com/DiedrichsenLab/cerebellar_atlases.git"
)
FWT_REPO = "https://github.com/KUL-Radneuron/KUL_FWT.git"
ELIAS2024_FIGSHARE_DOI = "https://doi.org/10.6084/m9.figshare.c.6844890.v1"
ELIAS2024_FIGSHARE_API = (
    "https://api.figshare.com/v2/collections/6844890/articles"
)

# ---------------------------------------------------------------------------
# Key atlas files expected after download
# ---------------------------------------------------------------------------

REQUIRED_ATLAS_FILES = [
    "atl-Anatom_space-SUIT_dseg.nii",
    "atl-Anatom_space-MNI_dseg.nii",
    "atl-Buckner7_space-SUIT_dseg.nii",
    "atl-Buckner17_space-SUIT_dseg.nii",
]

# Deep cerebellar nuclei file name patterns (case-insensitive search)
DCN_PATTERNS = [
    "dentate",
    "emboliform",
    "globose",
    "fastigial",
]

# Expected SUITPy bundled files (template & flatmap)
SUITPY_EXPECTED_FILES = [
    "SUIT.nii",
    "SUIT.surf.gii",
]

# Checksums output file
CHECKSUM_FILE = PROJECT_ROOT / "docs" / "data_sources_checksums.json"


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """
    Compute the SHA-256 hash of a file.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_checksum_registry() -> dict:
    """Load existing checksum registry from disk, or return an empty dict."""
    if CHECKSUM_FILE.exists():
        with open(CHECKSUM_FILE) as f:
            return json.load(f)
    return {}


def _save_checksum_registry(registry: dict) -> None:
    """Write the checksum registry to docs/data_sources_checksums.json."""
    CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKSUM_FILE, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    logger.info("Checksum registry saved: %s", CHECKSUM_FILE)


def _record_checksums(file_paths: list[Path], source_label: str) -> dict:
    """
    Compute SHA-256 checksums for a list of files and record them.

    Parameters
    ----------
    file_paths : list of Path
        Files to hash.
    source_label : str
        Label for this data source group (e.g. "cerebellar_atlases").

    Returns
    -------
    dict
        Mapping of relative file path (str) to checksum (str).
    """
    registry = _load_checksum_registry()
    if source_label not in registry:
        registry[source_label] = {}

    checksums = {}
    for fpath in file_paths:
        try:
            digest = sha256_file(fpath)
            rel = str(fpath.relative_to(PROJECT_ROOT))
            checksums[rel] = digest
            registry[source_label][rel] = {
                "sha256": digest,
                "size_bytes": fpath.stat().st_size,
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            }
            logger.info("  SHA-256 %s  %s", digest[:16] + "...", rel)
        except Exception as exc:
            logger.warning("  Could not hash %s: %s", fpath, exc)

    _save_checksum_registry(registry)
    return checksums


# ---------------------------------------------------------------------------
# Git clone helper
# ---------------------------------------------------------------------------

def clone_repo(url: str, dest: Path, shallow: bool = True) -> None:
    """
    Clone a git repository to *dest* using ``subprocess.run``.

    If *dest* already exists and is non-empty the clone is skipped.

    Parameters
    ----------
    url : str
        Remote repository URL.
    dest : Path
        Local destination directory.
    shallow : bool
        If True, clone with ``--depth 1`` to save bandwidth.
    """
    if dest.exists() and any(dest.iterdir()):
        logger.info("Directory already exists and is non-empty, skipping clone: %s", dest)
        return

    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    cmd.extend([url, str(dest)])

    logger.info("Cloning %s -> %s", url, dest)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Clone complete: %s", dest)
        if result.stdout.strip():
            logger.debug("git stdout: %s", result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        logger.error("git clone failed (return code %d)", exc.returncode)
        if exc.stderr:
            logger.error("git stderr: %s", exc.stderr.strip())
        raise RuntimeError(
            f"Failed to clone {url}. Check your network connection and that "
            f"git is installed."
        ) from exc


# ---------------------------------------------------------------------------
# HTTP download helper
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, chunk_size: int = 65536) -> None:
    """
    Download a file from *url* to *dest* with a tqdm progress bar.

    Skips the download if *dest* already exists.

    Parameters
    ----------
    url : str
        Remote URL.
    dest : Path
        Local file path.
    chunk_size : int
        Read chunk size in bytes.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("File already exists, skipping download: %s", dest)
        return

    logger.info("Downloading %s -> %s", url, dest)
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Network error downloading {url}. Check your internet connection."
        ) from exc
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"HTTP error {response.status_code} downloading {url}."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Timeout downloading {url}. Try again later."
        ) from exc

    total = int(response.headers.get("content-length", 0))

    with (
        open(dest, "wb") as f,
        tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Download complete: %s (%d bytes)", dest, dest.stat().st_size)


# ---------------------------------------------------------------------------
# Step 1: SUIT Cerebellar Atlases (Diedrichsen Lab)
# ---------------------------------------------------------------------------

def download_cerebellar_atlases() -> None:
    """
    Download the Diedrichsen lab cerebellar atlas collection.

    Clones ``https://github.com/DiedrichsenLab/cerebellar_atlases`` into
    ``data/raw/atlases/cerebellar_atlases/``.

    After cloning, verifies that the key parcellation NIfTI files are present
    and records their SHA-256 checksums.
    """
    logger.info("=" * 60)
    logger.info("Step 1: Downloading Cerebellar Atlases")
    logger.info("=" * 60)
    clone_repo(CEREBELLAR_ATLASES_REPO, ATLAS_DIR)

    # Verify key files exist (search recursively since repo layout may vary)
    found_files: list[Path] = []
    for fname in REQUIRED_ATLAS_FILES:
        matches = list(ATLAS_DIR.rglob(fname))
        if matches:
            logger.info("  Found: %s", matches[0].relative_to(ATLAS_DIR))
            found_files.append(matches[0])
        else:
            logger.warning("  MISSING: %s -- check repository structure", fname)

    # Record checksums for found atlas files
    if found_files:
        _record_checksums(found_files, "cerebellar_atlases")

    logger.info("Step 1 complete.")


# ---------------------------------------------------------------------------
# Step 2: Verify SUITPy installation (template + flatmap)
# ---------------------------------------------------------------------------

def verify_suit_installation() -> None:
    """
    Verify that SUITPy is installed and locate its bundled data files.

    SUITPy ships with the SUIT template (``SUIT.nii``), the flatmap surface
    (``SUIT.surf.gii``), and related definition files. This function checks
    that the package is importable and that these key files exist.
    """
    logger.info("=" * 60)
    logger.info("Step 2: Verifying SUITPy Installation")
    logger.info("=" * 60)

    try:
        import SUITPy  # noqa: F811
        suit_dir = Path(os.path.dirname(SUITPy.__file__))
        logger.info("SUITPy installed at: %s", suit_dir)
    except ImportError:
        logger.error(
            "SUITPy is NOT installed. Install with:\n"
            "  pip install git+https://github.com/DiedrichsenLab/SUITPy.git\n"
            "Then re-run this step."
        )
        return

    # Check for bundled data files
    data_dir = suit_dir / "data"
    found_files: list[Path] = []

    if data_dir.exists():
        logger.info("SUITPy data directory: %s", data_dir)
        for f in sorted(data_dir.iterdir()):
            logger.info("  %s  (%d bytes)", f.name, f.stat().st_size)
    else:
        logger.warning(
            "SUITPy data directory not found at expected location: %s", data_dir
        )

    # Search for specific expected files anywhere under the SUITPy tree
    for expected in SUITPY_EXPECTED_FILES:
        matches = list(suit_dir.rglob(expected))
        if matches:
            logger.info("  Found SUITPy file: %s", matches[0])
            found_files.append(matches[0])
        else:
            logger.warning("  MISSING SUITPy file: %s", expected)

    # Also search for flatmap-related files
    flatmap_files = list(suit_dir.rglob("*flat*"))
    if flatmap_files:
        logger.info("  Flatmap files found:")
        for f in flatmap_files:
            logger.info("    %s", f.name)
    else:
        logger.warning("  No flatmap files found in SUITPy package tree.")

    # Record checksums for SUITPy bundled data
    if found_files:
        _record_checksums(found_files, "suitpy_bundled")

    logger.info("Step 2 complete.")


# ---------------------------------------------------------------------------
# Step 3: FWT Atlas (Radwan et al. 2022)
# ---------------------------------------------------------------------------

def download_fwt_atlas() -> None:
    """
    Download the Radwan et al. (2022) Fun With Tracts white matter atlas.

    Clones ``https://github.com/KUL-Radneuron/KUL_FWT`` into
    ``data/raw/atlases/FWT/``.

    This atlas provides MNI-space probability maps for cerebellar peduncle
    tracts: SCP, DRTT, ICP, and MCP.
    """
    logger.info("=" * 60)
    logger.info("Step 3: Downloading FWT Atlas (Radwan et al. 2022)")
    logger.info("=" * 60)
    clone_repo(FWT_REPO, FWT_DIR)

    # Look for tract probability maps
    tract_files = sorted(FWT_DIR.rglob("*.nii*"))
    if tract_files:
        logger.info("Found %d NIfTI files in FWT atlas:", len(tract_files))
        for f in tract_files[:30]:
            logger.info("  %s", f.relative_to(FWT_DIR))
        if len(tract_files) > 30:
            logger.info("  ... and %d more", len(tract_files) - 30)

        # Record checksums for the NIfTI files
        _record_checksums(tract_files, "fwt_atlas")
    else:
        logger.warning(
            "No NIfTI files found in FWT atlas directory. "
            "Check repository structure at %s",
            FWT_DIR,
        )

    # Specifically look for cerebellar peduncle tracts
    peduncle_keywords = ["SCP", "scp", "DRTT", "drtt", "ICP", "icp", "MCP", "mcp"]
    peduncle_hits = [
        f
        for f in tract_files
        if any(kw in f.name for kw in peduncle_keywords)
    ]
    if peduncle_hits:
        logger.info("Cerebellar peduncle tract files found (%d):", len(peduncle_hits))
        for f in peduncle_hits:
            logger.info("  %s", f.relative_to(FWT_DIR))
    else:
        logger.info(
            "No files with SCP/DRTT/ICP/MCP in their names found directly. "
            "These may be in subdirectories -- check the FWT repo documentation."
        )

    logger.info("Step 3 complete.")


# ---------------------------------------------------------------------------
# Step 4: Elias et al. (2024) Normative Connectome (Figshare)
# ---------------------------------------------------------------------------

def download_normative_connectome() -> None:
    """
    Download / document the Elias et al. (2024) normative structural connectome.

    The connectome is a large dataset (multiple GB) hosted on Figshare at
    ``https://doi.org/10.6084/m9.figshare.c.6844890.v1``.

    This function:
    1. Checks whether data already exists in ``data/raw/hcp/elias2024_connectome/``.
    2. Attempts to query the Figshare API for available articles/files in the
       collection and lists them.
    3. If direct download links are found, downloads them with a progress bar.
    4. If the collection is too large for automated download, prints detailed
       manual-download instructions.
    """
    logger.info("=" * 60)
    logger.info("Step 4: Normative Connectome (Elias et al. 2024)")
    logger.info("=" * 60)
    logger.info("DOI: %s", ELIAS2024_FIGSHARE_DOI)

    HCP_DIR.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    existing = list(HCP_DIR.glob("*"))
    existing = [f for f in existing if not f.name.startswith(".")]
    if existing:
        logger.info(
            "Connectome directory already contains %d item(s):", len(existing)
        )
        for f in existing[:15]:
            size = f.stat().st_size if f.is_file() else 0
            logger.info("  %s  (%d bytes)", f.name, size)
        if len(existing) > 15:
            logger.info("  ... and %d more items", len(existing) - 15)

        # Record checksums for existing files
        file_list = [f for f in existing if f.is_file()]
        if file_list:
            _record_checksums(file_list, "elias2024_connectome")
        return

    # Attempt to query Figshare API for the collection contents
    logger.info("Querying Figshare API for collection contents...")
    articles = []
    try:
        resp = requests.get(ELIAS2024_FIGSHARE_API, timeout=30)
        resp.raise_for_status()
        articles = resp.json()
        logger.info(
            "Figshare collection contains %d article(s):", len(articles)
        )
        for art in articles:
            title = art.get("title", "untitled")
            art_id = art.get("id", "?")
            logger.info("  [%s] %s", art_id, title)
    except requests.RequestException as exc:
        logger.warning("Could not query Figshare API: %s", exc)

    # Attempt to download files from each article in the collection
    downloaded_files: list[Path] = []
    for art in articles:
        art_id = art.get("id")
        if not art_id:
            continue

        try:
            detail_url = f"https://api.figshare.com/v2/articles/{art_id}"
            detail_resp = requests.get(detail_url, timeout=30)
            detail_resp.raise_for_status()
            detail = detail_resp.json()

            files = detail.get("files", [])
            logger.info(
                "Article %s ('%s') has %d file(s):",
                art_id,
                detail.get("title", ""),
                len(files),
            )
            for finfo in files:
                fname = finfo.get("name", "unknown")
                fsize = finfo.get("size", 0)
                dl_url = finfo.get("download_url", "")
                logger.info(
                    "  %s  (%.1f MB)  %s",
                    fname,
                    fsize / 1e6,
                    dl_url[:80] if dl_url else "no URL",
                )

                if dl_url:
                    dest = HCP_DIR / fname
                    try:
                        download_file(dl_url, dest)
                        downloaded_files.append(dest)
                    except RuntimeError as dl_exc:
                        logger.warning(
                            "Failed to download %s: %s", fname, dl_exc
                        )
        except requests.RequestException as exc:
            logger.warning(
                "Could not fetch details for article %s: %s", art_id, exc
            )

    if downloaded_files:
        _record_checksums(downloaded_files, "elias2024_connectome")
        logger.info(
            "Downloaded %d file(s) to %s", len(downloaded_files), HCP_DIR
        )
    else:
        logger.info(
            "\n"
            "=========================================================\n"
            "  MANUAL DOWNLOAD REQUIRED\n"
            "=========================================================\n"
            "The normative connectome could not be downloaded automatically.\n"
            "This is a large dataset (multiple GB). To obtain it:\n"
            "\n"
            "  1. Visit: %s\n"
            "  2. Download all streamline files (.trk or .tck format)\n"
            "     and any accompanying metadata.\n"
            "  3. Place the downloaded files in:\n"
            "     %s\n"
            "  4. Re-run: python -m src.download --step checksums\n"
            "\n"
            "Alternatively, use the Figshare web interface or the\n"
            "figshare CLI tool (pip install figshare) to batch-download\n"
            "the entire collection.\n"
            "=========================================================",
            ELIAS2024_FIGSHARE_DOI,
            HCP_DIR,
        )

    logger.info("Step 4 complete.")


# ---------------------------------------------------------------------------
# Step 5: Verify deep cerebellar nuclei atlas files
# ---------------------------------------------------------------------------

def verify_dcn_atlases() -> None:
    """
    Verify that deep cerebellar nuclei (DCN) probability maps are present.

    Searches the cerebellar_atlases directory for probability maps of:
    - Dentate nucleus
    - Emboliform nucleus (anterior interposed)
    - Globose nucleus (posterior interposed)
    - Fastigial nucleus

    These may be individual NIfTI files or encoded as labels within a
    combined parcellation. Both forms are checked.
    """
    logger.info("=" * 60)
    logger.info("Step 5: Verifying Deep Cerebellar Nuclei Atlases")
    logger.info("=" * 60)

    if not ATLAS_DIR.exists():
        logger.error(
            "Cerebellar atlas directory does not exist: %s\n"
            "Run Step 1 (--step atlases) first.",
            ATLAS_DIR,
        )
        return

    found_nuclei: dict[str, list[Path]] = {}
    all_niftis = list(ATLAS_DIR.rglob("*.nii*"))

    for nucleus in DCN_PATTERNS:
        matches = [
            f
            for f in all_niftis
            if nucleus.lower() in f.name.lower()
        ]
        if matches:
            found_nuclei[nucleus] = matches
            logger.info(
                "  %s nucleus: %d file(s) found", nucleus.capitalize(), len(matches)
            )
            for m in matches:
                logger.info("    %s", m.relative_to(ATLAS_DIR))
        else:
            logger.warning(
                "  %s nucleus: no dedicated file found", nucleus.capitalize()
            )

    # Also check for a combined nuclei parcellation or probability atlas
    combined_candidates = [
        f
        for f in all_niftis
        if any(
            kw in f.name.lower()
            for kw in ["nuclei", "dcn", "deep", "interposed"]
        )
    ]
    if combined_candidates:
        logger.info("  Combined nuclei atlas candidates:")
        for c in combined_candidates:
            logger.info("    %s", c.relative_to(ATLAS_DIR))

    # Check the anatomical parcellation for nucleus labels
    anatom_files = list(ATLAS_DIR.rglob("*Anatom*SUIT*dseg*"))
    if anatom_files:
        logger.info(
            "  Note: The SUIT anatomical parcellation (%s) includes\n"
            "        nucleus labels (Dentate=29/30, Interposed=31/32, "
            "Fastigial=33/34).\n"
            "        These can be used as a fallback if dedicated probability\n"
            "        maps are not available.",
            anatom_files[0].name,
        )

    # Also look for .tsv label files that document nucleus labels
    tsv_files = list(ATLAS_DIR.rglob("*.tsv"))
    nuclei_tsvs = [
        f for f in tsv_files if any(kw in f.name.lower() for kw in ["nuclei", "dcn"])
    ]
    if nuclei_tsvs:
        logger.info("  DCN-related TSV label files:")
        for t in nuclei_tsvs:
            logger.info("    %s", t.relative_to(ATLAS_DIR))

    # Record checksums for any nucleus-specific NIfTI files
    nucleus_files = []
    for paths in found_nuclei.values():
        nucleus_files.extend(paths)
    if nucleus_files:
        _record_checksums(nucleus_files, "dcn_atlases")

    if not found_nuclei and not combined_candidates:
        logger.warning(
            "No deep cerebellar nuclei atlas files found.\n"
            "The model will fall back to nucleus labels within the SUIT\n"
            "anatomical parcellation (if available)."
        )

    logger.info("Step 5 complete.")


# ---------------------------------------------------------------------------
# Compute / refresh checksums for all downloaded data
# ---------------------------------------------------------------------------

def compute_all_checksums() -> None:
    """
    Walk all downloaded data directories and compute SHA-256 checksums.

    Results are written to ``docs/data_sources_checksums.json``.
    """
    logger.info("=" * 60)
    logger.info("Computing SHA-256 Checksums for All Data")
    logger.info("=" * 60)

    sources = {
        "cerebellar_atlases": ATLAS_DIR,
        "fwt_atlas": FWT_DIR,
        "elias2024_connectome": HCP_DIR,
    }

    for label, directory in sources.items():
        if not directory.exists():
            logger.info("  Skipping %s (directory does not exist)", label)
            continue

        nifti_files = sorted(directory.rglob("*.nii*"))
        trk_files = sorted(directory.rglob("*.trk"))
        tck_files = sorted(directory.rglob("*.tck"))
        all_files = nifti_files + trk_files + tck_files

        if not all_files:
            logger.info("  No data files found in %s", directory)
            continue

        logger.info(
            "  %s: hashing %d file(s)...", label, len(all_files)
        )
        _record_checksums(all_files, label)

    logger.info("All checksums written to %s", CHECKSUM_FILE)


# ---------------------------------------------------------------------------
# Verification: check all required data is present
# ---------------------------------------------------------------------------

def verify_all_data() -> bool:
    """
    Check whether all required data files are present on disk.

    Returns
    -------
    bool
        True if all required files are found, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("Verifying All Required Data")
    logger.info("=" * 60)

    all_ok = True

    # --- 1. Cerebellar atlases ---
    logger.info("Checking cerebellar atlases...")
    if not ATLAS_DIR.exists():
        logger.error("  MISSING: %s", ATLAS_DIR)
        all_ok = False
    else:
        for fname in REQUIRED_ATLAS_FILES:
            matches = list(ATLAS_DIR.rglob(fname))
            if matches:
                logger.info("  OK: %s", fname)
            else:
                logger.error("  MISSING: %s", fname)
                all_ok = False

    # --- 2. SUITPy ---
    logger.info("Checking SUITPy installation...")
    try:
        import SUITPy  # noqa: F811
        suit_dir = Path(os.path.dirname(SUITPy.__file__))
        for expected in SUITPY_EXPECTED_FILES:
            matches = list(suit_dir.rglob(expected))
            if matches:
                logger.info("  OK: %s", expected)
            else:
                logger.warning("  MISSING (SUITPy): %s", expected)
                # Not a hard failure -- SUITPy layout varies by version
    except ImportError:
        logger.error("  SUITPy is NOT installed.")
        all_ok = False

    # --- 3. FWT atlas ---
    logger.info("Checking FWT atlas...")
    if not FWT_DIR.exists():
        logger.error("  MISSING: %s", FWT_DIR)
        all_ok = False
    else:
        fwt_niis = list(FWT_DIR.rglob("*.nii*"))
        if fwt_niis:
            logger.info("  OK: %d NIfTI files in FWT", len(fwt_niis))
        else:
            logger.error("  MISSING: no NIfTI files in %s", FWT_DIR)
            all_ok = False

    # --- 4. Normative connectome ---
    logger.info("Checking normative connectome (Elias et al. 2024)...")
    if not HCP_DIR.exists():
        logger.warning("  MISSING: %s (manual download required)", HCP_DIR)
        all_ok = False
    else:
        hcp_files = [
            f for f in HCP_DIR.glob("*") if not f.name.startswith(".")
        ]
        if hcp_files:
            logger.info("  OK: %d file(s) in %s", len(hcp_files), HCP_DIR)
        else:
            logger.warning(
                "  EMPTY: %s -- see Step 4 for manual download instructions",
                HCP_DIR,
            )
            all_ok = False

    # --- 5. Deep cerebellar nuclei ---
    logger.info("Checking deep cerebellar nuclei atlases...")
    if ATLAS_DIR.exists():
        all_niftis = list(ATLAS_DIR.rglob("*.nii*"))
        for nucleus in DCN_PATTERNS:
            matches = [
                f for f in all_niftis if nucleus.lower() in f.name.lower()
            ]
            if matches:
                logger.info("  OK: %s nucleus (%d file(s))", nucleus, len(matches))
            else:
                # Check fallback: nucleus labels in the anatomical parcellation
                anatom = list(ATLAS_DIR.rglob("*Anatom*SUIT*dseg*"))
                if anatom:
                    logger.info(
                        "  INFO: %s -- no dedicated file, "
                        "but labels exist in anatomical parcellation",
                        nucleus,
                    )
                else:
                    logger.warning("  MISSING: %s nucleus", nucleus)
                    all_ok = False
    else:
        logger.error("  Cannot check DCN -- atlas directory missing.")
        all_ok = False

    # --- Summary ---
    if all_ok:
        logger.info("All required data files are present.")
    else:
        logger.warning(
            "Some required data files are missing. "
            "Run 'python -m src.download' to acquire them."
        )

    return all_ok


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate all data downloads for the cerebellar disconnectome project.

    Parses CLI arguments and dispatches to the appropriate download step(s).
    """
    parser = argparse.ArgumentParser(
        description="Download and verify data for the cerebellar disconnectome model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Steps:\n"
            "  all        Run all download and verification steps\n"
            "  atlases    Clone the Diedrichsen cerebellar atlas repo\n"
            "  suit       Verify SUITPy installation and bundled files\n"
            "  fwt        Clone the FWT white matter atlas repo\n"
            "  hcp        Download/document the Elias 2024 connectome\n"
            "  nuclei     Verify deep cerebellar nuclei atlas files\n"
            "  checksums  Recompute SHA-256 checksums for all data\n"
            "  verify     Check that all required data files exist\n"
        ),
    )
    parser.add_argument(
        "--step",
        choices=[
            "all",
            "atlases",
            "suit",
            "fwt",
            "hcp",
            "nuclei",
            "checksums",
            "verify",
        ],
        default="all",
        help="Which download / verification step to run (default: all).",
    )
    args = parser.parse_args()

    ensure_directories()

    step_map = {
        "atlases": download_cerebellar_atlases,
        "suit": verify_suit_installation,
        "fwt": download_fwt_atlas,
        "hcp": download_normative_connectome,
        "nuclei": verify_dcn_atlases,
        "checksums": compute_all_checksums,
        "verify": verify_all_data,
    }

    if args.step == "all":
        logger.info("Running all data acquisition steps...")
        download_cerebellar_atlases()
        verify_suit_installation()
        download_fwt_atlas()
        download_normative_connectome()
        verify_dcn_atlases()
        compute_all_checksums()
        verify_all_data()
        logger.info("=" * 60)
        logger.info("Data acquisition pipeline complete.")
        logger.info("Checksums recorded in: %s", CHECKSUM_FILE)
        logger.info("=" * 60)
    else:
        step_map[args.step]()


if __name__ == "__main__":
    main()
