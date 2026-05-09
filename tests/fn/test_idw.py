"""Tests for moirais.fn.idw — inverse distance weighting interpolation."""

import numpy as np
import pytest

from moirais.fn.idw import inverse_distance_weighting


class TestIDW:

    def test_exact_at_known_point(self):
        """Prediction at a known location returns the exact known value."""
        known_coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        known_values = np.array([10.0, 20.0, 30.0])
        predict_coords = np.array([[0, 0]], dtype=float)
        result = inverse_distance_weighting(known_coords, known_values,
                                            predict_coords, power=2.0)
        assert result.value[0] == pytest.approx(10.0)

    def test_midpoint_average(self):
        """Midpoint of two equal-distance points gives their average."""
        known_coords = np.array([[0, 0], [2, 0]], dtype=float)
        known_values = np.array([10.0, 30.0])
        predict_coords = np.array([[1, 0]], dtype=float)
        result = inverse_distance_weighting(known_coords, known_values,
                                            predict_coords, power=2.0)
        assert result.value[0] == pytest.approx(20.0, abs=0.01)

    def test_negative_power_raises(self):
        """Negative power raises ValueError."""
        with pytest.raises(ValueError, match="power must be"):
            inverse_distance_weighting(
                np.array([[0, 0]]), np.array([1.0]),
                np.array([[1, 1]]), power=-1.0)
