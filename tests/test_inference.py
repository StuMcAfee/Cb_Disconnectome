"""
Tests for the vertex-level inference engine.

Uses small synthetic efferent density maps, vertex projection matrices, and
binary lesion masks to verify the disruption inference pipeline.
"""

import numpy as np
import nibabel as nib
import pytest

from src.inference import infer_disruption, _detect_direct_injury
from src.utils import make_sphere_lesion


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOLUME_SHAPE = (10, 10, 10)
N_NUCLEI = 8  # Bilateral: 4 per hemisphere
N_VERTICES = 10
AFFINE = np.eye(4)  # 1mm isotropic, origin at (0, 0, 0)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data saved as temporary files
# ---------------------------------------------------------------------------

@pytest.fixture
def efferent_nifti(tmp_path):
    """
    Create a synthetic 4D efferent density volume (10x10x10x8).

    8 bilateral nuclei. Each has a distinct spatial profile:
      - nucleus 0 (left_fastigial):  gradient along x
      - nucleus 1 (left_emboliform): gradient along y
      - nucleus 2 (left_globose):    gradient along z
      - nucleus 3 (left_dentate):    uniform 0.5
      - nucleus 4 (right_fastigial): reverse gradient along x
      - nucleus 5 (right_emboliform): reverse gradient along y
      - nucleus 6 (right_globose):   reverse gradient along z
      - nucleus 7 (right_dentate):   uniform 0.3
    """
    data = np.zeros((*VOLUME_SHAPE, N_NUCLEI), dtype=np.float32)

    for i in range(VOLUME_SHAPE[0]):
        data[i, :, :, 0] = i / (VOLUME_SHAPE[0] - 1)
    for j in range(VOLUME_SHAPE[1]):
        data[:, j, :, 1] = j / (VOLUME_SHAPE[1] - 1)
    for k in range(VOLUME_SHAPE[2]):
        data[:, :, k, 2] = k / (VOLUME_SHAPE[2] - 1)
    data[:, :, :, 3] = 0.5
    for i in range(VOLUME_SHAPE[0]):
        data[i, :, :, 4] = 1.0 - i / (VOLUME_SHAPE[0] - 1)
    for j in range(VOLUME_SHAPE[1]):
        data[:, j, :, 5] = 1.0 - j / (VOLUME_SHAPE[1] - 1)
    for k in range(VOLUME_SHAPE[2]):
        data[:, :, k, 6] = 1.0 - k / (VOLUME_SHAPE[2] - 1)
    data[:, :, :, 7] = 0.3

    path = tmp_path / "efferent_4d.nii.gz"
    nib.save(nib.Nifti1Image(data, AFFINE), str(path))
    return path


@pytest.fixture
def vertex_projections(tmp_path):
    """
    Create synthetic vertex projections for 10 vertices (8 bilateral nuclei).

    Vertex projection profiles (8 elements: L_F, L_E, L_G, L_D, R_F, R_E, R_G, R_D):
      0: [1, 0, 0, 0, 0, 0, 0, 0]    (projects to left fastigial only)
      1: [0, 1, 0, 0, 0, 0, 0, 0]    (projects to left emboliform only)
      2: [0, 0, 1, 0, 0, 0, 0, 0]    (projects to left globose only)
      3: [0, 0, 0, 1, 0, 0, 0, 0]    (projects to left dentate only)
      4: [0.125]*8                     (uniform across all 8)
      5: [0.25, 0.25, 0, 0, 0.25, 0.25, 0, 0]  (fastigial+emboliform bilateral)
      6: [0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25]  (globose+dentate bilateral)
      7: [0.4, 0.05, 0.025, 0.025, 0.4, 0.05, 0.025, 0.025]  (mainly fastigial)
      8: [0, 0, 0, 0, 0, 0, 0, 0]    (zero projection edge case)
      9: [0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25]  (right side only)

    Vertex coordinates placed at known volume locations for direct
    injury testing.  Pial and white coords are offset slightly.
    """
    proj = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
        [0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25],
        [0.4, 0.05, 0.025, 0.025, 0.4, 0.05, 0.025, 0.025],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
    ], dtype=np.float32)

    # Place vertices at specific locations inside the 10x10x10 volume.
    # With identity affine, mm coords == voxel coords.
    pial_coords = np.array([
        [5.0, 5.0, 5.0],
        [3.0, 3.0, 3.0],
        [7.0, 7.0, 7.0],
        [1.0, 1.0, 1.0],
        [9.0, 9.0, 9.0],  # vertex 4: at corner
        [2.0, 5.0, 5.0],
        [5.0, 2.0, 5.0],
        [5.0, 5.0, 2.0],
        [0.0, 0.0, 0.0],
        [4.0, 4.0, 4.0],
    ], dtype=np.float32)

    # White coords slightly offset (simulating ~0.5mm cortical thickness)
    white_coords = pial_coords + 0.3

    parcel_labels = np.ones(N_VERTICES, dtype=np.int32)

    path = tmp_path / "vertex_projections.npz"
    np.savez(
        path,
        projections=proj,
        pial_coords=pial_coords,
        white_coords=white_coords,
        parcel_labels=parcel_labels,
    )
    return path


@pytest.fixture
def empty_lesion_nifti(tmp_path):
    """A lesion mask with all zeros (no lesion)."""
    data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    path = tmp_path / "empty_lesion.nii.gz"
    nib.save(nib.Nifti1Image(data, AFFINE), str(path))
    return path


@pytest.fixture
def single_voxel_lesion_nifti(tmp_path):
    """A lesion mask with a single voxel at position (9, 9, 9)."""
    data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    data[9, 9, 9] = 1.0
    path = tmp_path / "single_voxel_lesion.nii.gz"
    nib.save(nib.Nifti1Image(data, AFFINE), str(path))
    return path


@pytest.fixture
def block_lesion_nifti(tmp_path):
    """A lesion mask covering a 3x3x3 block at corner (7:10, 7:10, 7:10)."""
    data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    data[7:10, 7:10, 7:10] = 1.0
    path = tmp_path / "block_lesion.nii.gz"
    nib.save(nib.Nifti1Image(data, AFFINE), str(path))
    return path


@pytest.fixture
def cortical_lesion_nifti(tmp_path):
    """A lesion that overlaps vertex 4's location (at 9,9,9)."""
    data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    data[9, 9, 9] = 1.0
    path = tmp_path / "cortical_lesion.nii.gz"
    nib.save(nib.Nifti1Image(data, AFFINE), str(path))
    return path


@pytest.fixture
def mismatched_affine_lesion_nifti(tmp_path):
    """A lesion mask whose affine does NOT match the efferent volume."""
    data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    data[5, 5, 5] = 1.0
    bad_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    path = tmp_path / "mismatched_lesion.nii.gz"
    nib.save(nib.Nifti1Image(data, bad_affine), str(path))
    return path


# ---------------------------------------------------------------------------
# Tests: infer_disruption
# ---------------------------------------------------------------------------

class TestInferDisruption:
    """Tests for the core vertex-level disruption inference."""

    def test_infer_disruption_empty_lesion(
        self, efferent_nifti, vertex_projections, empty_lesion_nifti
    ):
        """An empty lesion should return all-zero disruption."""
        result = infer_disruption(
            empty_lesion_nifti, efferent_nifti, vertex_projections, method="max"
        )
        np.testing.assert_allclose(result, 0.0)
        assert result.shape == (N_VERTICES,)

    def test_infer_disruption_max_method(
        self, efferent_nifti, vertex_projections, single_voxel_lesion_nifti
    ):
        """
        Max method with a single voxel at (9,9,9).

        Efferent density at (9,9,9) for 8 bilateral nuclei:
          nucleus 0 (L_F): 9/9 = 1.0    nucleus 4 (R_F): 0.0
          nucleus 1 (L_E): 9/9 = 1.0    nucleus 5 (R_E): 0.0
          nucleus 2 (L_G): 9/9 = 1.0    nucleus 6 (R_G): 0.0
          nucleus 3 (L_D): 0.5           nucleus 7 (R_D): 0.3

        With only 1 lesion voxel, scores = E @ P.T where
        E = [1, 1, 1, 0.5, 0, 0, 0, 0.3]:
          vertex 0 (proj [1,0,0,0, 0,0,0,0]): 1.0
          vertex 3 (proj [0,0,0,1, 0,0,0,0]): 0.5
          vertex 8 (proj all zeros): 0.0
          vertex 9 (proj [0,0,0,0, .25,.25,.25,.25]):
              0*0.25 + 0*0.25 + 0*0.25 + 0.3*0.25 = 0.075
        """
        result = infer_disruption(
            single_voxel_lesion_nifti, efferent_nifti, vertex_projections,
            method="max",
        )
        assert result.shape == (N_VERTICES,)

        # Vertex 4 is at (9,9,9) so direct injury sets it to 1.0
        assert result[4] == pytest.approx(1.0)

        # Vertex 0: projects only to left fastigial (density=1.0)
        assert result[0] == pytest.approx(1.0, abs=1e-5)
        # Vertex 3: projects only to left dentate (density=0.5)
        assert result[3] == pytest.approx(0.5, abs=1e-5)
        # Vertex 8: zero projection -> 0.0
        assert result[8] == pytest.approx(0.0, abs=1e-5)
        # Vertex 9: projects only to right-side nuclei
        assert result[9] == pytest.approx(0.075, abs=1e-5)

    def test_infer_disruption_mean_method(
        self, efferent_nifti, vertex_projections, block_lesion_nifti
    ):
        """Mean method should average scores across the 27 lesion voxels."""
        result = infer_disruption(
            block_lesion_nifti, efferent_nifti, vertex_projections, method="mean"
        )
        assert result.shape == (N_VERTICES,)

        # Vertex 3 projects only to left dentate (index 3, uniform 0.5).
        # All lesion voxels have left dentate density = 0.5.
        # score for each voxel: 1 * 0.5 = 0.5, mean = 0.5
        assert result[3] == pytest.approx(0.5, abs=1e-5)

        # Vertex 8: zero projection -> 0.0
        assert result[8] == pytest.approx(0.0, abs=1e-5)

        # All values in [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-6)

    def test_infer_disruption_weighted_sum_method(
        self, efferent_nifti, vertex_projections, block_lesion_nifti
    ):
        """Weighted sum should be normalized to [0, 1]."""
        result = infer_disruption(
            block_lesion_nifti, efferent_nifti, vertex_projections,
            method="weighted_sum",
        )
        assert result.shape == (N_VERTICES,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-6)

    def test_infer_disruption_threshold_fraction_method(
        self, efferent_nifti, vertex_projections, block_lesion_nifti
    ):
        """Threshold fraction values should be in [0, 1]."""
        result = infer_disruption(
            block_lesion_nifti, efferent_nifti, vertex_projections,
            method="threshold_fraction",
        )
        assert result.shape == (N_VERTICES,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-6)

        # Vertex 8 has zero projection -> 0.0
        assert result[8] == pytest.approx(0.0, abs=1e-6)

    def test_infer_disruption_shape(
        self, efferent_nifti, vertex_projections, single_voxel_lesion_nifti
    ):
        """Output shape should match the number of vertices."""
        result = infer_disruption(
            single_voxel_lesion_nifti, efferent_nifti, vertex_projections,
            method="max",
        )
        assert result.shape == (N_VERTICES,)

    def test_infer_disruption_invalid_method(
        self, efferent_nifti, vertex_projections, single_voxel_lesion_nifti
    ):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            infer_disruption(
                single_voxel_lesion_nifti, efferent_nifti, vertex_projections,
                method="invalid",
            )


# ---------------------------------------------------------------------------
# Tests: direct injury detection
# ---------------------------------------------------------------------------

class TestDirectInjury:
    """Tests for direct cortical injury detection."""

    def test_direct_injury_vertex_overlap(self):
        """Vertex at lesion location should be detected as injured."""
        lesion_data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
        lesion_data[5, 5, 5] = 1.0

        pial_coords = np.array([
            [5.0, 5.0, 5.0],  # right on the lesion
            [0.0, 0.0, 0.0],  # far away
        ], dtype=np.float32)
        white_coords = pial_coords + 0.2

        injured = _detect_direct_injury(
            lesion_data, AFFINE, pial_coords, white_coords,
        )
        assert injured[0] == True
        assert injured[1] == False

    def test_direct_injury_no_lesion(self):
        """Empty lesion should not injure any vertex."""
        lesion_data = np.zeros(VOLUME_SHAPE, dtype=np.float32)

        pial_coords = np.array([
            [5.0, 5.0, 5.0],
            [3.0, 3.0, 3.0],
        ], dtype=np.float32)
        white_coords = pial_coords + 0.2

        injured = _detect_direct_injury(
            lesion_data, AFFINE, pial_coords, white_coords,
        )
        assert not injured.any()


# ---------------------------------------------------------------------------
# Tests: affine mismatch
# ---------------------------------------------------------------------------

class TestAffineMismatch:
    """Tests for spatial alignment validation."""

    def test_affine_mismatch_raises(
        self, efferent_nifti, vertex_projections, mismatched_affine_lesion_nifti
    ):
        """Mismatched affines between efferent and lesion should raise."""
        with pytest.raises(ValueError):
            infer_disruption(
                mismatched_affine_lesion_nifti,
                efferent_nifti,
                vertex_projections,
                method="max",
            )


# ---------------------------------------------------------------------------
# Tests: make_sphere_lesion (from src.utils — unchanged)
# ---------------------------------------------------------------------------

class TestMakeSphereLesion:
    """Tests for synthetic sphere lesion generation."""

    @pytest.fixture
    def reference_nifti(self, tmp_path):
        """Create a small reference NIfTI for sphere lesion generation."""
        data = np.zeros((20, 20, 20), dtype=np.float32)
        path = tmp_path / "reference.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
        return path

    def test_make_sphere_lesion_shape(self, reference_nifti):
        """Sphere lesion should have correct shape and be binary."""
        result_img = make_sphere_lesion((10.0, 10.0, 10.0), 3.0, reference_nifti)
        result_data = result_img.get_fdata()
        assert result_data.shape == (20, 20, 20)
        unique_vals = np.unique(result_data)
        assert all(v in [0.0, 1.0] for v in unique_vals)
        assert result_data.sum() > 0

    def test_make_sphere_lesion_center(self, reference_nifti):
        """The center voxel of the sphere should be non-zero."""
        result_img = make_sphere_lesion((10.0, 10.0, 10.0), 3.0, reference_nifti)
        assert result_img.get_fdata()[10, 10, 10] == 1.0

    def test_make_sphere_lesion_radius(self, reference_nifti):
        """Voxels far from center should be outside the sphere."""
        result_img = make_sphere_lesion((10.0, 10.0, 10.0), 2.0, reference_nifti)
        result_data = result_img.get_fdata()
        assert result_data[0, 0, 0] == 0.0
        assert result_data[10, 10, 13] == 0.0
