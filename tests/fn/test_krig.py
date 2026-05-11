"""Tests for morie.fn.krig — ordinary kriging."""

import numpy as np
import pytest

from morie.fn.krig import ordinary_kriging


class TestOrdinaryKriging:

    def test_predict_at_known_point(self):
        """Prediction at a known location is close to the known value."""
        known_coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        known_values = np.array([10.0, 20.0, 30.0, 40.0])
        predict_coords = np.array([[0, 0]], dtype=float)
        result = ordinary_kriging(known_coords, known_values, predict_coords,
                                  nugget=0.0, sill=100.0, range_=2.0)
        pred = result.value
        assert abs(pred[0] - 10.0) < 5.0, f"Expected ~10.0, got {pred[0]}"

    def test_output_length(self):
        """Output predictions match number of prediction points."""
        known_coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        known_values = np.array([1.0, 2.0, 3.0])
        predict_coords = np.array([[0.5, 0.5], [0.25, 0.25]], dtype=float)
        result = ordinary_kriging(known_coords, known_values, predict_coords,
                                  nugget=0.0, sill=1.0, range_=2.0)
        assert len(result.value) == 2
        assert "variances" in result.extra

    def test_shape_mismatch_raises(self):
        """Mismatched known_coords/known_values raises ValueError."""
        with pytest.raises(ValueError):
            ordinary_kriging(np.array([[0, 0]]), np.array([1.0, 2.0]),
                             np.array([[0, 0]]))
