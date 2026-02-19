"""
Tests for the cortico-nuclear mapping module.

Covers the zone classification logic (sigmoid weighting, medial-lateral
zone assignment) and the resulting nuclear target probabilities.
"""

import numpy as np
import pytest

from src.utils import sigmoid_weight, classify_zones, get_config


# ---------------------------------------------------------------------------
# sigmoid_weight
# ---------------------------------------------------------------------------

class TestSigmoidWeight:
    """Tests for the sigmoid boundary transition function."""

    def test_sigmoid_weight_center(self):
        """Sigmoid evaluated at zero (boundary center) should return ~0.5."""
        result = sigmoid_weight(np.array([0.0]), width=2.0)
        np.testing.assert_allclose(result, 0.5, atol=1e-7)

    def test_sigmoid_weight_extremes(self):
        """Far positive -> ~1.0, far negative -> ~0.0."""
        far_positive = sigmoid_weight(np.array([100.0]), width=2.0)
        far_negative = sigmoid_weight(np.array([-100.0]), width=2.0)

        np.testing.assert_allclose(far_positive, 1.0, atol=1e-6)
        np.testing.assert_allclose(far_negative, 0.0, atol=1e-6)

    def test_sigmoid_weight_symmetry(self):
        """sigmoid(d) + sigmoid(-d) should equal 1.0 (logistic symmetry)."""
        distances = np.array([-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
        forward = sigmoid_weight(distances, width=2.0)
        backward = sigmoid_weight(-distances, width=2.0)
        np.testing.assert_allclose(forward + backward, 1.0, atol=1e-12)

    def test_sigmoid_weight_monotonic(self):
        """Sigmoid should be monotonically increasing."""
        distances = np.linspace(-10.0, 10.0, 100)
        weights = sigmoid_weight(distances, width=2.0)
        assert np.all(np.diff(weights) >= 0)


# ---------------------------------------------------------------------------
# classify_zones
# ---------------------------------------------------------------------------

class TestClassifyZones:
    """Tests for the medial-lateral zone classification."""

    def test_classify_zones_midline(self):
        """x=0 should be classified as vermis with high probability."""
        x_mm = np.array([0.0])
        zones = classify_zones(x_mm)

        # At the midline, vermis weight should dominate
        assert zones["vermis"][0] > 0.9
        assert zones["paravermis"][0] < 0.1
        assert zones["lateral"][0] < 0.01

    def test_classify_zones_lateral(self):
        """Large |x| should be classified as lateral with high probability."""
        x_mm = np.array([50.0, -50.0])
        zones = classify_zones(x_mm)

        for i in range(2):
            assert zones["lateral"][i] > 0.9
            assert zones["vermis"][i] < 0.01
            assert zones["paravermis"][i] < 0.01

    def test_classify_zones_sum_to_one(self):
        """Zone weights must always sum to 1.0 for every voxel."""
        x_mm = np.linspace(-40.0, 40.0, 200)
        zones = classify_zones(x_mm)

        total = zones["vermis"] + zones["paravermis"] + zones["lateral"]
        np.testing.assert_allclose(total, 1.0, atol=1e-7)

    def test_classify_zones_custom_config(self):
        """Overriding threshold parameters should change zone assignments."""
        x_mm = np.array([8.0])

        # Default config: VERMIS_HALF_WIDTH_MM=5, PARAVERMIS_LATERAL_MM=15
        default_zones = classify_zones(x_mm)

        # Widen vermis to 12mm half-width -> x=8 should now be vermis
        wide_config = {"VERMIS_HALF_WIDTH_MM": 12.0}
        wide_zones = classify_zones(x_mm, config=wide_config)

        # With wider vermis, the vermis weight at x=8 should increase
        assert wide_zones["vermis"][0] > default_zones["vermis"][0]

    def test_classify_zones_paravermis_region(self):
        """A point between vermis and lateral boundaries should be paravermis."""
        # Default: vermis boundary at 5mm, lateral boundary at 15mm
        # x=10mm is squarely in the paravermis zone
        x_mm = np.array([10.0])
        zones = classify_zones(x_mm)

        assert zones["paravermis"][0] > 0.5
        assert zones["vermis"][0] < zones["paravermis"][0]
        assert zones["lateral"][0] < zones["paravermis"][0]

    def test_classify_zones_symmetry(self):
        """Zone weights should be symmetric around x=0."""
        x_values = np.array([3.0, 8.0, 20.0])
        zones_pos = classify_zones(x_values)
        zones_neg = classify_zones(-x_values)

        for zone_name in ["vermis", "paravermis", "lateral"]:
            np.testing.assert_allclose(
                zones_pos[zone_name], zones_neg[zone_name], atol=1e-12
            )


# ---------------------------------------------------------------------------
# Nuclear target probabilities (integration with zone classification)
# ---------------------------------------------------------------------------

class TestNuclearProbabilities:
    """
    Tests that verify the cortico-nuclear mapping produces sensible
    nuclear target probability distributions for different zone locations.

    These tests use the LOBULE_GROUPS table from cortico_nuclear_map to
    check that zone-weighted nuclear probabilities have the expected
    dominant nucleus.
    """

    def test_nuclear_probabilities_vermis(self):
        """Vermis voxels should have high fastigial probability.

        For any lobule group, the vermis row assigns the highest probability
        to the fastigial nucleus (index 0). We verify this by checking that
        a midline voxel's zone weights, when applied to the mapping table,
        produce fastigial as the dominant target.
        """
        from src.cortico_nuclear_map import LOBULE_GROUPS

        x_mm = np.array([0.0])  # midline = vermis
        zones = classify_zones(x_mm)

        # Test across all lobule groups
        for group_key, group_info in LOBULE_GROUPS.items():
            prob = np.zeros(4)
            for zone_name in ["vermis", "paravermis", "lateral"]:
                prob += zones[zone_name][0] * np.array(group_info[zone_name])

            # Normalize
            prob = prob / prob.sum()

            # Fastigial (index 0) should be the dominant nucleus for vermis
            assert prob[0] == prob.max(), (
                f"Expected fastigial to dominate for vermis in group "
                f"'{group_key}', got probabilities {prob}"
            )

    def test_nuclear_probabilities_lateral(self):
        """Lateral voxels should have high dentate probability.

        For most lobule groups, the lateral row assigns the highest probability
        to the dentate nucleus (index 3).
        """
        from src.cortico_nuclear_map import LOBULE_GROUPS

        x_mm = np.array([50.0])  # far lateral
        zones = classify_zones(x_mm)

        for group_key, group_info in LOBULE_GROUPS.items():
            prob = np.zeros(4)
            for zone_name in ["vermis", "paravermis", "lateral"]:
                prob += zones[zone_name][0] * np.array(group_info[zone_name])

            prob = prob / prob.sum()

            # Dentate (index 3) should be the dominant nucleus for lateral
            assert prob[3] == prob.max(), (
                f"Expected dentate to dominate for lateral in group "
                f"'{group_key}', got probabilities {prob}"
            )

    def test_nuclear_probabilities_normalization(self):
        """Nuclear probabilities must sum to 1.0 for any zone mixture.

        We test a range of x coordinates covering vermis through lateral
        and all lobule groups to confirm normalization.
        """
        from src.cortico_nuclear_map import LOBULE_GROUPS

        x_values = np.linspace(-40.0, 40.0, 50)
        zones = classify_zones(x_values)

        for group_key, group_info in LOBULE_GROUPS.items():
            for idx in range(len(x_values)):
                prob = np.zeros(4)
                for zone_name in ["vermis", "paravermis", "lateral"]:
                    prob += zones[zone_name][idx] * np.array(
                        group_info[zone_name]
                    )

                # Each group's zone rows already sum to 1, and zone weights
                # sum to 1, so the combined probabilities should sum to 1
                np.testing.assert_allclose(
                    prob.sum(), 1.0, atol=1e-6,
                    err_msg=(
                        f"Probabilities do not sum to 1 for group "
                        f"'{group_key}' at x={x_values[idx]:.1f}mm"
                    ),
                )
