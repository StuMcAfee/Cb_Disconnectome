# Cerebellar Efferent Disconnectome Model

A computational model that takes a binary lesion mask in SUIT cerebellar space and infers which cerebellar cortical domains are functionally disrupted — both by direct cortical injury and by downstream disconnection of efferent pathways through the deep cerebellar nuclei and superior cerebellar peduncle (SCP).

## Motivation

The cerebellar efferent pathway flows:

**Cerebellar cortex → Deep cerebellar nuclei → Superior cerebellar peduncle → (decussation) → Red nucleus / Thalamus → Cerebral cortex**

A lesion anywhere along this pathway disconnects the upstream cortical regions whose output funnels through the damaged segment. This convergence is especially pronounced at the SCP, where output from broad cortical territories narrows into a compact fiber bundle. A small SCP lesion can therefore produce widespread cortical disconnection — a pattern not captured by traditional lesion-symptom mapping that only considers direct cortical damage.

This tool produces **vertex-level cortical disruption maps** at the resolution of the SUIT cerebellar surface mesh (28,935 vertices). At inference time, a lesion mask is combined with per-vertex nuclear projection probabilities and efferent density maps via an on-the-fly matrix multiplication, producing a disruption score for each surface vertex that maps directly onto the SUIT flatmap.

## Installation

### Requirements

- Python >= 3.10
- Git LFS (for large neuroimaging files)

### Setup

```bash
# Clone the repository
git clone https://github.com/StuMcAfee/Cb_Disconnectome.git
cd Cb_Disconnectome

# Install SUITPy (not on PyPI)
pip install git+https://github.com/DiedrichsenLab/SUITPy.git

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

### Data Acquisition

Download all required atlas and connectome data:

```bash
python -m src.download
```

This downloads:
- SUIT cerebellar atlas collection (lobular parcellations, nuclei probability maps)
- Radwan et al. (2022) Fun With Tracts white matter atlas
- Elias et al. (2024) normative structural connectome (large download)

See [docs/data_sources.md](docs/data_sources.md) for full provenance information.

## Quick Start

### Run inference on a lesion mask

```bash
python -m src.inference path/to/lesion_SUIT.nii.gz [output_directory]
```

### Python API

```python
from src.inference import infer_disruption, infer_disruption_sparse
from src.flatmap import plot_disruption_flatmap

# Compute vertex-level disruption probabilities (28,935 values)
disruption = infer_disruption(
    'my_lesion_SUIT.nii.gz',
    method='max'
)

# Or use the sparse matrix backend (captures all disconnection layers)
disruption = infer_disruption_sparse(
    'my_lesion_SUIT.nii.gz',
    method='max'
)

# Render directly on SUIT flatmap (no volume projection needed)
plot_disruption_flatmap(
    disruption,
    lesion_data='my_lesion_SUIT.nii.gz',
    output_path='my_disruption_flatmap.png'
)
```

## Pipeline Overview

The model is built in 6 steps:

| Step | Module | Description |
|------|--------|-------------|
| 0 | `src/download.py` | Download atlas and connectome data |
| 1 | `src/cortico_nuclear_map.py` | Map cerebellar cortex voxels to deep nuclei targets |
| 2 | `src/tractography.py` | Build efferent pathway density maps from normative tractography |
| 3 | `src/build_4d_nifti.py` | Compute per-vertex nuclear projections using SUIT surfaces |
| 3b | `src/disconnection_matrix.py` | Build sparse voxel-to-vertex disconnection matrix |
| 4 | `src/inference.py` | Lesion mask → vertex-wise cortical disruption probabilities |
| 5 | `src/flatmap.py` | Render results on SUIT cerebellar flatmap |

### Build the full pipeline

```bash
# Step 0: Download data
python -m src.download

# Step 1: Build cortico-nuclear projection map
python -m src.cortico_nuclear_map

# Step 2: Build efferent pathway density maps
python -m src.tractography

# Step 3: Compute per-vertex nuclear projections
python -m src.build_4d_nifti

# Step 3b: Build sparse disconnection matrix (optional, enables sparse inference)
python -m src.disconnection_matrix
```

## Vertex-Level Architecture

The core data products are:

### Step 3 outputs (in `data/final/`)

- **`vertex_projections.npz`** — per-vertex bilateral nuclear projection data:
  - `projections` (28935, 8): probability that each vertex projects to each bilateral nucleus
  - `pial_coords` (28935, 3): pial surface coordinates in SUIT mm
  - `white_coords` (28935, 3): white-matter surface coordinates in SUIT mm
  - `parcel_labels` (28935,): SUIT lobular label per vertex (1-28)
- **`efferent_density_suit_4d.nii.gz`** — combined efferent density maps (X, Y, Z, 8)
- **`vertex_metadata.json`** — JSON metadata sidecar

Bilateral nuclei (8 total, 4 per hemisphere):
- Indices 0-3: left fastigial, left emboliform, left globose, left dentate
- Indices 4-7: right fastigial, right emboliform, right globose, right dentate

### Step 3b outputs (in `data/final/`)

- **`disconnection_matrix.npz`** — sparse CSR matrix (n_suit_voxels, 28935):
  `D[v, p]` = probability that lesioning voxel `v` disconnects vertex `p`.
  Captures four disconnection mechanisms:
  - **Layer 1:** Direct cortical injury (cortical voxels overlapping vertex locations)
  - **Layers 2+3:** Nuclear relay + efferent pathway (via `e @ P.T` for each voxel)
  - **Layer 4:** Internal WM disconnection (placeholder for future implementation)
- **`disconnection_matrix_metadata.json`** — JSON metadata sidecar

### Inference

Two inference backends are available:

**On-the-fly matmul** (default): For a lesion mask with N voxels:

1. Extract efferent density at lesion voxels → `E` of shape `(N_lesion, 8)`
2. Matrix multiply: `E @ P.T` → `(N_lesion, 28935)` per-voxel per-vertex scores
3. Aggregate across lesion voxels (max/mean/etc.) → `(28935,)` disruption scores
4. Direct injury: vertices whose cortical locations overlap the lesion → set to 1.0

**Sparse matrix** (captures all disconnection layers): For a lesion mask:

1. Select lesion rows from the precomputed sparse matrix → `D_lesion` (N_lesion, 28935)
2. Aggregate across lesion voxels (max/mean/weighted_sum) → `(28935,)` disruption scores

The sparse backend is recommended when the disconnection matrix has been built
(Step 3b) because it captures all disconnection mechanisms in a single lookup.

Both backends output vertex-indexed arrays that map directly onto the SUIT flatmap — no volume-to-surface projection is needed.

## Inference Methods

Four aggregation methods are available:

| Method | Formula | Interpretation |
|--------|---------|----------------|
| `max` (default) | max weighted density across lesion voxels | "Weakest link" — single-voxel transection suffices |
| `mean` | average weighted density across lesion voxels | Smoothed estimate weighted by spatial extent |
| `weighted_sum` | normalized sum of weighted densities | Correlates with lesion size |
| `threshold_fraction` | per-nucleus pathway fraction, combined by projection | Most biologically interpretable |

The default is `max`, following the principle that a single point of complete fiber transection is sufficient to disconnect all streamlines passing through it.

## Modeling Assumptions

This model relies on several anatomical assumptions documented in [docs/assumptions.md](docs/assumptions.md). Key assumptions include:

- **A1:** Cortico-nuclear mapping follows the Voogd medial-lateral zonal scheme
- **A2:** Lobule-to-zone assignment uses configurable medial-lateral thresholds
- **A3:** Nuclear-to-SCP topographic continuity is preserved
- **A4:** Normative (healthy adult) tractography approximates patient anatomy

These assumptions and their limitations are discussed in detail in the assumptions document.

## Project Structure

```
├── src/                    # Core pipeline modules
│   ├── download.py         # Data acquisition
│   ├── cortico_nuclear_map.py  # Cortex-to-nuclei mapping
│   ├── tractography.py     # Efferent pathway modeling
│   ├── build_4d_nifti.py   # Vertex projection assembly
│   ├── disconnection_matrix.py  # Sparse voxel-to-vertex disconnection matrix
│   ├── inference.py        # Lesion inference engine
│   ├── flatmap.py          # SUIT flatmap visualization
│   └── utils.py            # Shared utilities
├── qa/                     # Quality assurance scripts
├── tests/                  # Unit tests
├── docs/                   # Documentation and QA reports
├── data/                   # Data directory (raw/interim/final)
└── notebooks/              # Jupyter notebooks
```

## References

- Diedrichsen, J. (2006). A spatially unbiased atlas template of the human cerebellum. *NeuroImage*, 33(1), 127-138.
- Diedrichsen, J., et al. (2009). A probabilistic MR atlas of the human cerebellum. *NeuroImage*, 46(1), 39-46.
- Diedrichsen, J., et al. (2011). Imaging the deep cerebellar nuclei: a probabilistic atlas and normalization procedure. *NeuroImage*, 54(3), 1786-1794.
- Elias, G.J.B., et al. (2024). Normative connectomes and their use in DBS. *Scientific Data*, 11, 390.
- Radwan, A.M., et al. (2022). An atlas of white matter anatomy. *NeuroImage*, 254, 119029.
- Apps, R., & Garwicz, M. (2005). Anatomical and physiological foundations of cerebellar information processing. *Nature Reviews Neuroscience*, 6(4), 297-311.
- Voogd, J. (1967). Comparative aspects of the structure and fibre connexions of the mammalian cerebellum. *Progress in Brain Research*, 25, 94-134.

## License

MIT License. See [LICENSE](LICENSE) for details.
