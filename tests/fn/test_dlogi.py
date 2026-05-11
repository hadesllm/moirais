"""Tests for morie.fn.dlogi -- Logistic PDF."""

import numpy as np
import pytest
from morie.fn.dlogi import dlogi


class TestDlogi:
    def test_at_zero(self):
        """dlogi(0) = 0.25 for standard logistic."""
        assert dlogi(0) == pytest.approx(0.25, abs=1e-10)

    def test_symmetry(self):
        """Standard logistic is symmetric: dlogi(-x) == dlogi(x)."""
        assert dlogi(-2.0) == pytest.approx(dlogi(2.0), abs=1e-12)

    def test_nonstandard(self):
        """dlogi(loc, loc, scale) = 1/(4*scale)."""
        assert dlogi(3.0, loc=3.0, scale=2.0) == pytest.approx(1.0 / (4 * 2), abs=1e-10)

    def test_array(self):
        result = dlogi(np.array([-1, 0, 1]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_raises_bad_scale(self):
        with pytest.raises(ValueError):
            dlogi(0, scale=0)
