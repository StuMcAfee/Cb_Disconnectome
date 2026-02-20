"""
Tests for the sparse disconnection matrix builder and sparse inference.
"""

import numpy as np
import nibabel as nib
import pytest
import scipy.sparse as sp

from src.disconnection_matrix import (
    _build_cortical_layer,
    _build_efferent_layer,
    _build_wm_layer,
    combine_layers,
)
from src.inference import infer_disruption_sparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOLUME_SHAPE = (10, 10, 10)
N_NUCLEI = 8
N_VERTICES = 10
N_TOTAL_VOXELS = 10 * 10 * 10
AFFINE = np.eye(4)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vertex_coords():
    """Pial and white coordinates for 10 test vertices."""
    pial = np.array([
        [5.0, 5.0, 5.0],
        [3.0, 3.0, 3.0],
        [7.0, 7.0, 7.0],
        [1.0, 1.0, 1.0],
        [9.0, 9.0, 9.0],
        [2.0, 5.0, 5.0],
        [5.0, 2.0, 5.0],
        [5.0, 5.0, 2.0],
        [0.0, 0.0, 0.0],
        [4.0, 4.0, 4.0],
    ], dtype=np.float64)
    white = pial + 0.3
    return pial, white


@pytest.fixture
def vertex_projections_8():
    """Vertex projection matrix (10 vertices, 8 bilateral nuclei)."""
    return np.array([
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


@pytest.fixture
def efferent_data_8():
    """Synthetic 4D efferent density (10x10x10x8)."""
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
    return data


# ---------------------------------------------------------------------------
# Tests: Layer 1 (cortical injury)
# ---------------------------------------------------------------------------

class TestCorticalLayer:
    """Tests for the direct cortical injury layer."""

    def test_cortical_layer_shape(self, vertex_coords):
        """Output should be sparse matrix of correct shape."""
        pial, white = vertex_coords
        D1 = _build_cortical_layer(pial, white, AFFINE, VOLUME_SHAPE)
        assert D1.shape == (N_TOTAL_VOXELS, N_VERTICES)
        assert sp.issparse(D1)

    def test_cortical_layer_nonzero(self, vertex_coords):
        """Should have nonzero entries for vertex locations."""
        pial, white = vertex_coords
        D1 = _build_cortical_layer(pial, white, AFFINE, VOLUME_SHAPE)
        assert D1.nnz > 0

    def test_cortical_layer_values_capped(self, vertex_coords):
        """All values should be at most 1.0."""
        pial, white = vertex_coords
        D1 = _build_cortical_layer(pial, white, AFFINE, VOLUME_SHAPE)
        assert D1.max() <= 1.0
        assert D1.min() >= 0.0

    def test_cortical_layer_vertex_at_voxel(self, vertex_coords):
        """Vertex 0 at (5,5,5) should be affected by voxel (5,5,5)."""
        pial, white = vertex_coords
        D1 = _build_cortical_layer(pial, white, AFFINE, VOLUME_SHAPE)

        flat_idx = np.ravel_multi_index((5, 5, 5), VOLUME_SHAPE)
        assert D1[flat_idx, 0] > 0


# ---------------------------------------------------------------------------
# Tests: Layers 2+3 (efferent pathway)
# ---------------------------------------------------------------------------

class TestEfferentLayer:
    """Tests for the nuclear + efferent pathway layer."""

    def test_efferent_layer_shape(self, efferent_data_8, vertex_projections_8):
        """Output should be sparse matrix of correct shape."""
        D23 = _build_efferent_layer(
            efferent_data_8, vertex_projections_8, VOLUME_SHAPE,
        )
        assert D23.shape == (N_TOTAL_VOXELS, N_VERTICES)
        assert sp.issparse(D23)

    def test_efferent_layer_nonzero(self, efferent_data_8, vertex_projections_8):
        """Should have nonzero entries for voxels with efferent density."""
        D23 = _build_efferent_layer(
            efferent_data_8, vertex_projections_8, VOLUME_SHAPE,
        )
        assert D23.nnz > 0

    def test_efferent_layer_zero_projection_vertex(
        self, efferent_data_8, vertex_projections_8,
    ):
        """Vertex 8 (zero projection) should have all-zero column."""
        D23 = _build_efferent_layer(
            efferent_data_8, vertex_projections_8, VOLUME_SHAPE,
        )
        col_8 = D23[:, 8].toarray().ravel()
        assert np.all(col_8 == 0)

    def test_efferent_layer_values_reasonable(
        self, efferent_data_8, vertex_projections_8,
    ):
        """All values should be non-negative."""
        D23 = _build_efferent_layer(
            efferent_data_8, vertex_projections_8, VOLUME_SHAPE,
        )
        assert D23.min() >= 0.0


# ---------------------------------------------------------------------------
# Tests: Layer 4 (WM placeholder)
# ---------------------------------------------------------------------------

class TestWMLayer:
    """Tests for the WM disconnection placeholder layer."""

    def test_wm_layer_empty(self):
        """Placeholder should return an empty sparse matrix."""
        D4 = _build_wm_layer(VOLUME_SHAPE, N_VERTICES)
        assert D4.shape == (N_TOTAL_VOXELS, N_VERTICES)
        assert D4.nnz == 0


# ---------------------------------------------------------------------------
# Tests: Layer combination
# ---------------------------------------------------------------------------

class TestCombineLayers:
    """Tests for combining disconnection layers."""

    def test_combine_takes_max(self):
        """Combined matrix should take element-wise max across layers."""
        n_vox, n_vert = 100, 10

        D1 = sp.csr_matrix(
            (np.array([0.3, 0.8]), (np.array([0, 5]), np.array([0, 3]))),
            shape=(n_vox, n_vert),
        )
        D23 = sp.csr_matrix(
            (np.array([0.5, 0.4]), (np.array([0, 5]), np.array([0, 3]))),
            shape=(n_vox, n_vert),
        )
        D4 = sp.csr_matrix((n_vox, n_vert), dtype=np.float32)

        D = combine_layers(D1, D23, D4)

        # max(0.3, 0.5) = 0.5 at (0, 0)
        assert D[0, 0] == pytest.approx(0.5)
        # max(0.8, 0.4) = 0.8 at (5, 3)
        assert D[5, 3] == pytest.approx(0.8)

    def test_combine_clips_to_one(self):
        """Values above 1.0 should be clipped."""
        n_vox, n_vert = 10, 5
        D1 = sp.csr_matrix(
            (np.array([1.5]), (np.array([0]), np.array([0]))),
            shape=(n_vox, n_vert),
        )
        D23 = sp.csr_matrix((n_vox, n_vert), dtype=np.float32)
        D4 = sp.csr_matrix((n_vox, n_vert), dtype=np.float32)

        D = combine_layers(D1, D23, D4)
        assert D[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: Sparse inference
# ---------------------------------------------------------------------------

class TestSparseInference:
    """Tests for sparse matrix-based inference."""

    @pytest.fixture
    def sparse_matrix_npz(self, tmp_path, vertex_coords, vertex_projections_8, efferent_data_8):
        """Build and save a sparse disconnection matrix for testing."""
        pial, white = vertex_coords
        D1 = _build_cortical_layer(pial, white, AFFINE, VOLUME_SHAPE)
        D23 = _build_efferent_layer(
            efferent_data_8, vertex_projections_8, VOLUME_SHAPE,
        )
        D4 = _build_wm_layer(VOLUME_SHAPE, N_VERTICES)
        D = combine_layers(D1, D23, D4)

        path = tmp_path / "disconnection_matrix.npz"
        sp.save_npz(path, D)
        return path

    @pytest.fixture
    def efferent_nifti(self, tmp_path, efferent_data_8):
        """Save efferent data as NIfTI for reference."""
        path = tmp_path / "efferent_4d.nii.gz"
        nib.save(nib.Nifti1Image(efferent_data_8, AFFINE), str(path))
        return path

    @pytest.fixture
    def projections_npz(self, tmp_path, vertex_coords, vertex_projections_8):
        """Save vertex projections."""
        pial, white = vertex_coords
        path = tmp_path / "vertex_projections.npz"
        np.savez(
            path,
            projections=vertex_projections_8,
            pial_coords=pial.astype(np.float32),
            white_coords=white.astype(np.float32),
            parcel_labels=np.ones(N_VERTICES, dtype=np.int32),
        )
        return path

    def test_sparse_inference_empty_lesion(
        self, tmp_path, sparse_matrix_npz, efferent_nifti, projections_npz,
    ):
        """Empty lesion should return all-zero disruption."""
        lesion_data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
        lesion_path = tmp_path / "empty_lesion.nii.gz"
        nib.save(nib.Nifti1Image(lesion_data, AFFINE), str(lesion_path))

        result = infer_disruption_sparse(
            lesion_path, sparse_matrix_npz, efferent_nifti, projections_npz,
            method="max",
        )
        assert result.shape == (N_VERTICES,)
        np.testing.assert_allclose(result, 0.0)

    def test_sparse_inference_single_voxel(
        self, tmp_path, sparse_matrix_npz, efferent_nifti, projections_npz,
    ):
        """Single voxel lesion should produce nonzero disruption."""
        lesion_data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
        lesion_data[5, 5, 5] = 1.0
        lesion_path = tmp_path / "single_lesion.nii.gz"
        nib.save(nib.Nifti1Image(lesion_data, AFFINE), str(lesion_path))

        result = infer_disruption_sparse(
            lesion_path, sparse_matrix_npz, efferent_nifti, projections_npz,
            method="max",
        )
        assert result.shape == (N_VERTICES,)
        # Voxel (5,5,5) has efferent density and overlaps vertex 0
        assert result[0] > 0

    def test_sparse_inference_values_in_range(
        self, tmp_path, sparse_matrix_npz, efferent_nifti, projections_npz,
    ):
        """All disruption values should be in [0, 1]."""
        lesion_data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
        lesion_data[7:10, 7:10, 7:10] = 1.0
        lesion_path = tmp_path / "block_lesion.nii.gz"
        nib.save(nib.Nifti1Image(lesion_data, AFFINE), str(lesion_path))

        result = infer_disruption_sparse(
            lesion_path, sparse_matrix_npz, efferent_nifti, projections_npz,
            method="max",
        )
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-6)

    def test_sparse_inference_zero_projection_vertex(
        self, tmp_path, sparse_matrix_npz, efferent_nifti, projections_npz,
    ):
        """Vertex 8 (zero projection, not at any lesion) should remain 0."""
        # Place lesion far from vertex 8's location (0,0,0)
        lesion_data = np.zeros(VOLUME_SHAPE, dtype=np.float32)
        lesion_data[5, 5, 5] = 1.0
        lesion_path = tmp_path / "away_lesion.nii.gz"
        nib.save(nib.Nifti1Image(lesion_data, AFFINE), str(lesion_path))

        result = infer_disruption_sparse(
            lesion_path, sparse_matrix_npz, efferent_nifti, projections_npz,
            method="max",
        )
        assert result[8] == pytest.approx(0.0, abs=1e-6)
