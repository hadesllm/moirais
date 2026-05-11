"""Tests for morie.fn.gstat — empirical semivariogram."""

import numpy as np
import pytest

from morie.fn.gstat import semivariogram


class TestSemivariogram:

    def test_semivariance_increases_with_lag(self):
        """Spatially autocorrelated data shows increasing semivariance."""
        rng = np.random.default_rng(42)
        n = 50
        coords = rng.uniform(0, 10, size=(n, 2))
        values = coords[:, 0] + 0.1 * rng.standard_normal(n)
        result = semivariogram(coords, values, n_bins=10)
        sv = result.extra["semivariance"]
        counts = result.extra["counts"]
        filled = counts > 0
        sv_filled = sv[filled]
        assert len(sv_filled) >= 3
        assert sv_filled[-1] > sv_filled[0], "Semivariance should increase with lag"

    def test_output_shapes(self):
        """Output arrays have correct lengths."""
        coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = semivariogram(coords, values, n_bins=5)
        assert len(result.extra["lags"]) == 5
        assert len(result.extra["semivariance"]) == 5
        assert len(result.extra["counts"]) == 5

    def test_shape_mismatch_raises(self):
        """Mismatched coords/values raises ValueError."""
        with pytest.raises(ValueError):
            semivariogram(np.array([[0, 0], [1, 1]]), np.array([1.0, 2.0, 3.0]))
