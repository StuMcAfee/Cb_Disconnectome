"""
Tests for the inference engine.

Uses small synthetic 4D occupancy volumes and binary lesion masks saved as
temporary NIfTI files to verify the disruption inference pipeline.
"""

import numpy as np
import nibabel as nib
import pytest

from src.inference import infer_disruption, disruption_to_volume
from src.utils import make_sphere_lesion


# ---------------------------------------------------------------------------
# Fixtures: synthetic data saved as temporary NIfTI files
# ---------------------------------------------------------------------------

VOLUME_SHAPE = (10, 10, 10)
N_PARCELS = 5
AFFINE = np.eye(4)  # 1mm isotropic, origin at (0, 0, 0)


@pytest.fixture
def occupancy_nifti(tmp_path):
    """
    Create a synthetic 4D occupancy volume (10x10x10x5).

    Each parcel's occupancy map is a gradient along a different axis so
    the parcels have distinct spatial profiles:
      - parcel 0: gradient along x
      - parcel 1: gradient along y
      - parcel 2: gradient along z
      - parcel 3: uniform 0.5
      - parcel 4: uniform 0.0 (empty pathway)
    """
    data = np.zeros((*VOLUME_SHAPE, N_PARCELS), dtype=np.float32)

    # Parcel 0: linearly increasing along x, range [0, 1]
    for i in range(VOLUME_SHAPE[0]):
        data[i, :, :, 0] = i / (VOLUME_SHAPE[0] - 1)

    # Parcel 1: linearly increasing along y
    for j in range(VOLUME_SHAPE[1]):
        data[:, j, :, 1] = j / (VOLUME_SHAPE[1] - 1)

    # Parcel 2: linearly increasing along z
    for k in range(VOLUME_SHAPE[2]):
        data[:, :, k, 2] = k / (VOLUME_SHAPE[2] - 1)

    # Parcel 3: uniform 0.5
    data[:, :, :, 3] = 0.5

    # Parcel 4: all zeros (no pathway)
    # Already initialized to 0

    path = tmp_path / "occupancy.nii.gz"
    img = nib.Nifti1Image(data, AFFINE)
    nib.save(img, str(path))
    return path


@pytest.fixture
def empty_lesion_nifti(tmp_path):
    """A lesion mask with all zeros (no lesion)."""
    data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    path = tmp_path / "empty_lesion.nii.gz"
    img = nib.Nifti1Image(data, AFFINE)
    nib.save(img, str(path))
    return path


@pytest.fixture
def single_voxel_lesion_nifti(tmp_path):
    """A lesion mask with a single voxel at position (9, 9, 9)."""
    data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    data[9, 9, 9] = 1.0
    path = tmp_path / "single_voxel_lesion.nii.gz"
    img = nib.Nifti1Image(data, AFFINE)
    nib.save(img, str(path))
    return path


@pytest.fixture
def block_lesion_nifti(tmp_path):
    """A lesion mask covering a 3x3x3 block at corner (7:10, 7:10, 7:10)."""
    data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    data[7:10, 7:10, 7:10] = 1.0
    path = tmp_path / "block_lesion.nii.gz"
    img = nib.Nifti1Image(data, AFFINE)
    nib.save(img, str(path))
    return path


@pytest.fixture
def mismatched_affine_lesion_nifti(tmp_path):
    """A lesion mask whose affine does NOT match the occupancy volume."""
    data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    data[5, 5, 5] = 1.0
    bad_affine = np.diag([2.0, 2.0, 2.0, 1.0])  # 2mm isotropic
    path = tmp_path / "mismatched_lesion.nii.gz"
    img = nib.Nifti1Image(data, bad_affine)
    nib.save(img, str(path))
    return path


# ---------------------------------------------------------------------------
# Tests: infer_disruption
# ---------------------------------------------------------------------------

class TestInferDisruption:
    """Tests for the core disruption inference function."""

    def test_infer_disruption_empty_lesion(
        self, occupancy_nifti, empty_lesion_nifti
    ):
        """An empty lesion (all zeros) should return all-zero disruption."""
        result = infer_disruption(
            empty_lesion_nifti, occupancy_nifti, method="max"
        )
        np.testing.assert_allclose(result, 0.0)

    def test_infer_disruption_max_method(
        self, occupancy_nifti, single_voxel_lesion_nifti
    ):
        """
        Max method with single voxel at (9,9,9) should return the occupancy
        values at that location.

        For our synthetic data:
          parcel 0 at x=9: 9/9 = 1.0
          parcel 1 at y=9: 9/9 = 1.0
          parcel 2 at z=9: 9/9 = 1.0
          parcel 3: 0.5
          parcel 4: 0.0
        """
        result = infer_disruption(
            single_voxel_lesion_nifti, occupancy_nifti, method="max"
        )
        expected = np.array([1.0, 1.0, 1.0, 0.5, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_infer_disruption_mean_method(
        self, occupancy_nifti, block_lesion_nifti
    ):
        """
        Mean method should return the average occupancy across lesion voxels.

        The 3x3x3 block at (7:10, 7:10, 7:10) contains 27 voxels.
        For parcel 0 (x-gradient): x values are 7,8,9 -> occ = 7/9, 8/9, 9/9
        averaged over the 3 x-slices (y and z don't matter for parcel 0):
        mean = (7+8+9) / (3*9) = 24/27 = 8/9 ~ 0.8889
        """
        result = infer_disruption(
            block_lesion_nifti, occupancy_nifti, method="mean"
        )

        assert result.shape == (N_PARCELS,)

        # Parcel 0: mean of x-gradient over block
        expected_p0 = np.mean([i / 9.0 for i in [7, 8, 9]])
        np.testing.assert_allclose(result[0], expected_p0, atol=1e-5)

        # Parcel 3: uniform 0.5, mean should be 0.5
        np.testing.assert_allclose(result[3], 0.5, atol=1e-5)

        # Parcel 4: all zeros
        np.testing.assert_allclose(result[4], 0.0, atol=1e-5)

    def test_infer_disruption_weighted_sum_method(
        self, occupancy_nifti, block_lesion_nifti
    ):
        """
        Weighted sum should be normalized to [0, 1].

        The weighted sum accumulates occupancy across all lesion voxels,
        then normalizes so the result falls in [0, 1].
        """
        result = infer_disruption(
            block_lesion_nifti, occupancy_nifti, method="weighted_sum"
        )

        assert result.shape == (N_PARCELS,)

        # All values must be in [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-7)

        # Parcel 4 (empty pathway) should still be 0
        np.testing.assert_allclose(result[4], 0.0, atol=1e-7)

    def test_infer_disruption_threshold_fraction_method(
        self, occupancy_nifti, block_lesion_nifti
    ):
        """
        Threshold fraction method returns the fraction of the pathway
        that is intersected by the lesion.

        For each parcel, this is the number of lesion voxels where
        occupancy > 0 divided by the total number of voxels where
        occupancy > 0 in the whole volume.
        """
        result = infer_disruption(
            block_lesion_nifti, occupancy_nifti, method="threshold_fraction"
        )

        assert result.shape == (N_PARCELS,)

        # All values must be in [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-7)

        # Parcel 4 (empty pathway) should be 0
        np.testing.assert_allclose(result[4], 0.0, atol=1e-7)

        # Parcel 3 (uniform 0.5 everywhere): 27 lesion voxels out of 1000 total
        # that have occupancy > 0 (all 1000 voxels have occ = 0.5 > 0)
        expected_p3 = 27.0 / (10 * 10 * 10)
        np.testing.assert_allclose(result[3], expected_p3, atol=1e-5)

    def test_infer_disruption_shape(
        self, occupancy_nifti, single_voxel_lesion_nifti
    ):
        """Output shape should match the number of parcels."""
        result = infer_disruption(
            single_voxel_lesion_nifti, occupancy_nifti, method="max"
        )
        assert result.shape == (N_PARCELS,)


# ---------------------------------------------------------------------------
# Tests: disruption_to_volume
# ---------------------------------------------------------------------------

class TestDisruptionToVolume:
    """Tests for mapping disruption scores back to a spatial volume."""

    def test_disruption_to_volume_shape(self, tmp_path):
        """Output volume should have the correct 3D spatial shape."""
        # Create a simple parcellation where each voxel has a label 1-5
        parc_data = np.zeros(VOLUME_SHAPE, dtype=np.int32)
        for i in range(N_PARCELS):
            parc_data[i * 2 : (i + 1) * 2, :, :] = i + 1

        parc_path = tmp_path / "parcellation.nii.gz"
        nib.save(nib.Nifti1Image(parc_data, AFFINE), str(parc_path))

        ref_path = tmp_path / "reference.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(VOLUME_SHAPE, dtype=np.float32), AFFINE), str(ref_path))

        disruption_scores = np.array([0.8, 0.6, 0.4, 0.2, 0.0])
        result_img = disruption_to_volume(disruption_scores, parc_path, ref_path)

        # Result should be a NIfTI image with the correct 3D spatial shape
        assert result_img.shape == VOLUME_SHAPE


# ---------------------------------------------------------------------------
# Tests: affine mismatch
# ---------------------------------------------------------------------------

class TestAffineMismatch:
    """Tests for spatial alignment validation."""

    def test_affine_mismatch_raises(
        self, occupancy_nifti, mismatched_affine_lesion_nifti
    ):
        """Mismatched affines between occupancy and lesion should raise."""
        with pytest.raises(ValueError):
            infer_disruption(
                mismatched_affine_lesion_nifti,
                occupancy_nifti,
                method="max",
            )


# ---------------------------------------------------------------------------
# Tests: make_sphere_lesion (from src.utils)
# ---------------------------------------------------------------------------

class TestMakeSphereLesion:
    """Tests for synthetic sphere lesion generation."""

    @pytest.fixture
    def reference_nifti(self, tmp_path):
        """Create a small reference NIfTI for sphere lesion generation."""
        data = np.zeros((20, 20, 20), dtype=np.float32)
        path = tmp_path / "reference.nii.gz"
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(path))
        return path

    def test_make_sphere_lesion_shape(self, reference_nifti):
        """Sphere lesion should have correct shape and be binary."""
        center_mm = (10.0, 10.0, 10.0)
        radius_mm = 3.0

        result_img = make_sphere_lesion(center_mm, radius_mm, reference_nifti)
        result_data = result_img.get_fdata()

        # Shape matches reference
        assert result_data.shape == (20, 20, 20)

        # Binary: only 0s and 1s
        unique_vals = np.unique(result_data)
        assert all(v in [0.0, 1.0] for v in unique_vals)

        # There should be some nonzero voxels (the sphere is inside the volume)
        assert result_data.sum() > 0

    def test_make_sphere_lesion_center(self, reference_nifti):
        """The center voxel of the sphere should be non-zero."""
        center_mm = (10.0, 10.0, 10.0)
        radius_mm = 3.0

        result_img = make_sphere_lesion(center_mm, radius_mm, reference_nifti)
        result_data = result_img.get_fdata()

        # With identity affine, mm coords = voxel coords
        # The center voxel at (10, 10, 10) should be inside the sphere
        assert result_data[10, 10, 10] == 1.0

    def test_make_sphere_lesion_radius(self, reference_nifti):
        """Voxels far from center should be outside the sphere."""
        center_mm = (10.0, 10.0, 10.0)
        radius_mm = 2.0

        result_img = make_sphere_lesion(center_mm, radius_mm, reference_nifti)
        result_data = result_img.get_fdata()

        # A corner voxel far from center should be 0
        assert result_data[0, 0, 0] == 0.0

        # A voxel just outside the radius should be 0
        # Distance from (10,10,10) to (10,10,13) = 3mm > 2mm radius
        assert result_data[10, 10, 13] == 0.0
