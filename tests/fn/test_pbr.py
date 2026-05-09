"""Tests for moirais.fn.pbr -- Point-biserial correlation."""

import numpy as np
import pytest
from moirais.fn.pbr import point_biserial_r


class TestPointBiserialR:
    def test_known_correlation(self):
        """Binary grouping with different means yields nonzero r."""
        binary = [0]*20 + [1]*20
        continuous = [1.0]*20 + [5.0]*20
        result = point_biserial_r(binary, continuous)
        assert isinstance(result, dict)
        assert "r" in result
        assert abs(result["r"]) > 0.5

    def test_no_separation_near_zero(self, rng):
        """Random binary and continuous should give r near 0."""
        binary = rng.choice([0, 1], 200)
        continuous = rng.standard_normal(200)
        result = point_biserial_r(binary, continuous)
        assert abs(result["r"]) < 0.3

    def test_p_value_present(self):
        """Result dict should contain p_value."""
        result = point_biserial_r([0, 0, 1, 1], [1.0, 2.0, 3.0, 4.0])
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1
