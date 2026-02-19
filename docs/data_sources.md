# Data Sources

All datasets used in this project, with provenance, licensing, and access information.

## 1. SUIT Cerebellar Atlas Collection

**Citation:** Diedrichsen, J. (2006). A spatially unbiased atlas template of the
human cerebellum. NeuroImage, 33(1), 127-138.

**URL:** https://github.com/DiedrichsenLab/cerebellar_atlases

**License:** CC-BY-4.0

**Files used:**
- `atl-Anatom_space-SUIT_dseg.nii` — Lobular parcellation in SUIT space
- `atl-Anatom_space-MNI_dseg.nii` — Lobular parcellation in MNI space
- `atl-Buckner7_space-SUIT_dseg.nii` — Buckner 7-network parcellation
- `atl-Buckner17_space-SUIT_dseg.nii` — Buckner 17-network parcellation
- Deep cerebellar nuclei probability maps (dentate, interposed, fastigial)

**Download location:** `data/raw/atlases/cerebellar_atlases/`

## 2. SUITPy (SUIT Python Toolbox)

**Citation:** Diedrichsen, J., Balsters, J.H., Flavell, J., Cussans, E., &
Ramnani, N. (2009). A probabilistic MR atlas of the human cerebellum.
NeuroImage, 46(1), 39-46.

**URL:** https://github.com/DiedrichsenLab/SUITPy

**License:** MIT

**Files used (bundled with package):**
- `SUIT.nii` — SUIT template
- `SUIT.surf.gii` — Flatmap surface geometry
- Flatmap definition files

**Installation:** `pip install git+https://github.com/DiedrichsenLab/SUITPy.git`

## 3. Radwan et al. (2022) Fun With Tracts (FWT) Atlas

**Citation:** Radwan, A.M., Sunaert, S., Schilling, K., et al. (2022). An atlas
of white matter anatomy, its variability, and reproducibility based on constrained
spherical deconvolution of diffusion MRI. NeuroImage, 254, 119029.

**URL:** https://github.com/KUL-Radneuron/KUL_FWT

**License:** Apache-2.0

**Files used:**
- SCP (Superior Cerebellar Peduncle) probability maps in MNI space
- DRTT (Dentato-Rubro-Thalamic Tract) probability maps
- ICP and MCP probability maps

**Download location:** `data/raw/atlases/FWT/`

## 4. Elias et al. (2024) Normative Structural Connectome

**Citation:** Elias, G.J.B., et al. (2024). Normative connectomes and their
use in DBS. Scientific Data, 11, 390. doi:10.1038/s41597-024-03197-0

**DOI:** https://doi.org/10.6084/m9.figshare.c.6844890.v1

**License:** CC-BY-4.0

**Description:** Precomputed whole-brain streamlines in MNI space derived from
a large cohort of healthy adults using multi-shell diffusion MRI from the
Human Connectome Project.

**Download location:** `data/raw/hcp/elias2024_connectome/`

**Note:** This is a very large dataset (multiple GB). Confirm available disk
space before downloading. This was chosen over running tractography from scratch
(Option C) or using Lead-DBS connectomes (Option B) for reproducibility and
ease of setup.

## 5. SUIT Deep Cerebellar Nuclei Atlas

**Citation:** Diedrichsen, J., Maderwald, S., Kuper, M., et al. (2011).
Imaging the deep cerebellar nuclei: a probabilistic atlas and normalization
procedure. NeuroImage, 54(3), 1786-1794.

**URL:** Part of the cerebellar atlas collection (see item 1)

**Files used:**
- Dentate nucleus probability map
- Emboliform (anterior interposed) nucleus probability map
- Globose (posterior interposed) nucleus probability map
- Fastigial nucleus probability map

**Space:** SUIT

## Checksums

After downloading, verify file integrity:

```bash
# Run from project root after download
sha256sum data/raw/atlases/cerebellar_atlases/atl-Anatom_space-SUIT_dseg.nii
sha256sum data/raw/atlases/cerebellar_atlases/atl-Buckner7_space-SUIT_dseg.nii
# ... (populate after first download)
```

Checksums will be recorded here after initial data acquisition.
