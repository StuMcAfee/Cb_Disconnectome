# Cerebellar Efferent Disconnectome Model

A computational model that takes a binary lesion mask in SUIT cerebellar space and infers which cerebellar cortical domains are functionally disrupted — both by direct cortical injury and by downstream disconnection of efferent pathways through the deep cerebellar nuclei and superior cerebellar peduncle (SCP).

## Motivation

The cerebellar efferent pathway flows:

**Cerebellar cortex → Deep cerebellar nuclei → Superior cerebellar peduncle → (decussation) → Red nucleus / Thalamus → Cerebral cortex**

A lesion anywhere along this pathway disconnects the upstream cortical regions whose output funnels through the damaged segment. This convergence is especially pronounced at the SCP, where output from broad cortical territories narrows into a compact fiber bundle. A small SCP lesion can therefore produce widespread cortical disconnection — a pattern not captured by traditional lesion-symptom mapping that only considers direct cortical damage.

This tool produces a **4D NIfTI "pathway occupancy volume"** where each 3D volume represents the spatial probability that efferent streamlines from one cerebellar cortical parcel pass through each white matter voxel. At inference time, intersecting a lesion mask with this 4D volume produces a cortical disruption map that can be projected onto the SUIT cerebellar flatmap.

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
from src.inference import infer_disruption, disruption_to_volume
from src.flatmap import plot_disruption_flatmap

# Compute disruption probabilities
disruption = infer_disruption(
    'my_lesion_SUIT.nii.gz',
    'data/final/pathway_occupancy_4d.nii.gz',
    method='max'
)

# Convert to volume for visualization
vol = disruption_to_volume(
    disruption,
    'data/final/parcellation_subdivided_SUIT.nii.gz',
    'data/final/pathway_occupancy_4d.nii.gz'
)
vol.to_filename('my_disruption_map.nii.gz')

# Render on SUIT flatmap
plot_disruption_flatmap(
    'my_disruption_map.nii.gz',
    lesion_volume_path='my_lesion_SUIT.nii.gz',
    output_path='my_disruption_flatmap.png'
)
```

## Pipeline Overview

The model is built in 5 steps:

| Step | Module | Description |
|------|--------|-------------|
| 0 | `src/download.py` | Download atlas and connectome data |
| 1 | `src/cortico_nuclear_map.py` | Map cerebellar cortex voxels to deep nuclei targets |
| 2 | `src/tractography.py` | Build efferent pathway density maps from normative tractography |
| 3 | `src/build_4d_nifti.py` | Assemble the 4D pathway occupancy volume |
| 4 | `src/inference.py` | Lesion mask → cortical disruption probabilities |
| 5 | `src/flatmap.py` | Project results onto SUIT cerebellar flatmap |

### Build the full pipeline

```bash
# Step 0: Download data
python -m src.download

# Step 1: Build cortico-nuclear projection map
python -m src.cortico_nuclear_map

# Step 2: Build efferent pathway density maps
python -m src.tractography

# Step 3: Assemble 4D pathway occupancy volume
python -m src.build_4d_nifti

# Run QA checks
python qa/qa_01_atlas_alignment.py
python qa/qa_02_cortico_nuclear.py
python qa/qa_03_tractography.py
python qa/qa_04_4d_nifti.py
python qa/qa_05_inference_sanity.py
python qa/qa_06_flatmap_projection.py
```

## The 4D Pathway Occupancy Volume

The primary output is `data/final/pathway_occupancy_4d.nii.gz`:

- **Space:** SUIT cerebellar space
- **Shape:** `(X, Y, Z, N_parcels)` where N_parcels ≈ 60–80
- **Values:** Probability [0, 1] that a lesion at each voxel disrupts each cortical parcel
  - `1.0` = direct cortical injury (the voxel is within the parcel)
  - `0 < p < 1` = downstream disconnection via efferent pathways
  - `0.0` = no relationship

Parcel definitions and metadata are in `data/final/pathway_occupancy_metadata.json`.

## Inference Methods

Four aggregation methods are available:

| Method | Formula | Interpretation |
|--------|---------|----------------|
| `max` | max occupancy across lesion voxels | "Weakest link" — single-voxel transection suffices |
| `mean` | average occupancy across lesion voxels | Smoothed estimate; underestimates focal lesions |
| `weighted_sum` | normalized sum of occupancy | Correlates with lesion size |
| `threshold_fraction` | fraction of pathway volume intersected | Most biologically interpretable |

The default is `max`, following the principle that a single point of complete fiber transection is sufficient to disconnect all streamlines passing through it.

## Modeling Assumptions

This model relies on several anatomical assumptions documented in [docs/assumptions.md](docs/assumptions.md). Key assumptions include:

- **A1:** Cortico-nuclear mapping follows the Voogd medial-lateral zonal scheme
- **A2:** Lobule-to-zone assignment uses configurable medial-lateral thresholds
- **A3:** Nuclear-to-SCP topographic continuity is preserved
- **A4:** Normative (healthy adult) tractography approximates patient anatomy

These assumptions and their limitations are discussed in detail in the assumptions document.

## Quality Assurance

Six QA scripts validate each pipeline stage. Reports are saved to `docs/qa_reports/`. See the [QA checklist](#global-quality-assurance-checklist) for expected results.

## Project Structure

```
├── src/                    # Core pipeline modules
│   ├── download.py         # Data acquisition
│   ├── cortico_nuclear_map.py  # Cortex-to-nuclei mapping
│   ├── tractography.py     # Efferent pathway modeling
│   ├── build_4d_nifti.py   # 4D volume assembly
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
