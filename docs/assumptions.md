# Modeling Assumptions

## A1: Cortico-Nuclear Mapping Based on Voogd Zonal Scheme (Bilateral)

**Assumption:** The corticonuclear projection in humans follows the medial-lateral
zonal scheme established in animal models (Voogd 1967; Apps & Garwicz 2005;
Sugihara & Shinoda 2007). Projections are **ipsilateral**: left cerebellar cortex
projects to left deep nuclei, right cortex to right nuclei. Vermis cortex
projects bilaterally (split equally).

**Mapping rules applied:**

| Cerebellar cortical region | Target deep nucleus | Confidence |
|---|---|---|
| Vermis (lobules I-X midline) | Bilateral fastigial | High |
| Paravermal lobules I-V | Ipsilateral emboliform (ant. interposed) | Moderate |
| Paravermal lobules VI-IX | Ipsilateral globose (post. interposed) | Moderate |
| Lateral hemisphere, anterior lobe | Ipsilateral dentate (dorsal/motor domain) | High |
| Lateral hemisphere, Crus I/II | Ipsilateral dentate (ventral/nonmotor domain) | High |
| Lateral hemisphere, lobules VIIb-IX | Ipsilateral dentate (intermediate) | Moderate |
| Flocculonodular lobe (lobule X) | Bilateral fastigial + vestibular nuclei | High |

**Bilateral representation:** The model uses 8 nuclear channels (4 per hemisphere):
left/right × {fastigial, emboliform, globose, dentate}. Laterality is determined
from the SUIT atlas label names (Left/Right/Vermis).

**Confidence levels:**
- **High:** supported by convergent evidence from multiple animal species AND human
  functional connectivity data
- **Moderate:** supported by animal tracing data; human evidence is indirect
  (functional connectivity or clinical correlation)
- **Low:** inferred by anatomical analogy; minimal direct evidence

**Limitations:**
- The zonal boundaries in humans are not directly observable on MRI and are
  approximated from lobular boundaries
- Convergence within the nuclei means a single nuclear voxel may receive input
  from multiple cortical zones — the mapping is probabilistic, not one-to-one
- The interposed nuclei are very small in humans and difficult to resolve even
  at high field strengths; their probability maps have wide spatial uncertainty

## A2: Lobule-to-Zone Assignment

**Assumption:** SUIT lobular parcellations can be assigned to Voogd zones using
the following medial-lateral segmentation:

- **Vermis:** voxels within +/-5mm of midline (x=0 in SUIT space)
  - This threshold is a parameter (`VERMIS_HALF_WIDTH_MM`) and should be
    tested at 3mm, 5mm, and 7mm
- **Paravermal:** voxels between the vermis boundary and 15mm lateral
  - This threshold is a parameter (`PARAVERMIS_LATERAL_MM`)
- **Lateral hemisphere:** voxels beyond the paravermal boundary

**Limitation:** These boundaries are inherently continuous, not discrete. The
true zones are defined by molecular markers (aldolase C / zebrin) that are not
visible on MRI. The model outputs should be interpreted as probabilistic
estimates, not sharp categorical assignments.

## A3: Nuclear-to-SCP Topographic Continuity

**Assumption:** The topographic organization within the deep cerebellar nuclei
is preserved as fibers enter and traverse the superior cerebellar peduncle. That
is, fibers from adjacent nuclear territories travel in adjacent positions within
the SCP.

**Evidence:** This is supported by the Meola et al. (2016) finding that the
nondecussating DRTT occupies a distinct position within the SCP relative to
the decussating DRTT, and by tract-tracing studies showing topographic
organization of SCP fibers (van Baarsen et al. 2016).

**Limitation:** The degree of fiber intermingling within the SCP is not fully
characterized in humans. The model assumes complete topographic segregation,
which likely overestimates the specificity of downstream disconnection
predictions.

## A4: Normative Tractography Represents Patient Anatomy

**Assumption:** Streamlines from the normative connectome (healthy adults)
provide a reasonable approximation of white matter organization in the patient
population.

**Limitation:** This is weakest for:
- Pediatric patients (white matter is still maturing)
- Patients with large tumors that displace or infiltrate white matter
- Patients with prior surgery or radiation

The model should be validated against patient-specific diffusion data when
available.

## A5: Sigmoid Blending at Zone Boundaries

**Assumption:** The transition between medial-lateral zones (vermis, paravermis,
lateral hemisphere) is gradual, not sharp. We model this using a sigmoid function
with a 2mm transition width.

**Rationale:** The biological zone boundaries are defined by molecular expression
gradients (zebrin/aldolase C stripes) that are inherently smooth. A hard threshold
would introduce artifactual discontinuities in the disruption predictions.

**Parameter:** `BOUNDARY_TRANSITION_WIDTH_MM = 2.0`

## A6: Inference Aggregation Method

**Assumption:** The default inference method is `max` — for each cortical surface
vertex, the disruption probability equals the maximum weighted efferent density
across all lesion voxels.

**Rationale:** A single voxel of complete fiber transection is sufficient to
disconnect all streamlines passing through that point. The `max` method captures
this "weakest link" principle.

**Alternative methods** (implemented for comparison):
- `mean`: average weighted density across lesion voxels (smoothed estimate)
- `weighted_sum`: total weighted density affected (correlates with lesion size)
- `threshold_fraction`: per-nucleus pathway fraction, combined by vertex projection weights

## A7: Vertex-Level Resolution via SUIT Surfaces

**Assumption:** Cortical disruption is resolved at the level of individual SUIT
surface vertices (28,935 vertices) rather than at the level of whole lobular
parcels (28 parcels).

**Rationale:** The SUIT toolbox provides matched grey-matter (pial) and
white-matter surface meshes in 3D space, plus a flat surface for 2D visualization.
All three surfaces share identical vertex indexing, so a value computed at vertex N
on the 3D surface can be directly plotted at vertex N on the flatmap.

The cortico-nuclear probability map from Step 1 is sampled at each vertex's
cortical location (6 depths between pial and white surfaces), preserving the
smooth medial-lateral gradients rather than collapsing them into parcel averages.

**Implementation:** Two inference backends are available:

1. **On-the-fly matmul:** `E_lesion @ P.T` where `E_lesion` is (N_lesion, 8) and
   `P` is (28935, 8). Avoids precomputing the full occupancy volume.
2. **Sparse disconnection matrix:** A precomputed sparse CSR matrix `D` of shape
   (n_suit_voxels, 28935) that encodes all disconnection mechanisms. At inference,
   lesion rows are selected and aggregated directly.

## A8: Sparse Disconnection Matrix Layers

**Assumption:** The total disconnection effect of a voxel on a cortical vertex
can be decomposed into four independent layers, combined via element-wise maximum:

| Layer | Mechanism | Status |
|-------|-----------|--------|
| 1 | **Direct cortical injury**: voxel overlaps vertex cortical location | Implemented |
| 2+3 | **Nuclear + efferent pathway**: voxel disrupts nuclear relay or SCP fibers | Implemented |
| 4 | **Internal WM (arbor vitae)**: voxel severs cortex-to-nucleus fibers | Placeholder |

**Rationale:** Each layer captures a distinct anatomical mechanism by which a
lesion can disconnect cerebellar cortex. The element-wise max combination reflects
that any single mechanism is sufficient for disconnection.

**Limitation:** Layer 4 (internal WM) is currently a placeholder returning zero.
This means lesions restricted to the cerebellar white matter between cortex and
nuclei will not produce cortex-to-nucleus disconnection predictions. Implementing
Layer 4 requires either:
- Tractography-based modeling: extracting cortex-to-nucleus streamlines from the
  normative connectome
- Geometric modeling: radial projection from cortex through WM to nuclei based
  on the folial architecture of the cerebellum
