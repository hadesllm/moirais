"""Tests for moirais.fn.dcchy -- Cauchy PDF."""

import numpy as np
import pytest
from moirais.fn.dcchy import dcchy


class TestDcchy:
    def test_at_zero(self):
        """dcchy(0) = 1/pi ~ 0.31831."""
        assert dcchy(0) == pytest.approx(1.0 / np.pi, abs=1e-5)

    def test_symmetry(self):
        """Standard Cauchy is symmetric: dcchy(-x) == dcchy(x)."""
        assert dcchy(-2.0) == pytest.approx(dcchy(2.0), abs=1e-12)

    def test_nonstandard(self):
        """dcchy(loc, loc=loc, scale=s) = 1/(pi*s)."""
        assert dcchy(3.0, loc=3.0, scale=2.0) == pytest.approx(1.0 / (np.pi * 2), abs=1e-5)

    def test_array(self):
        result = dcchy(np.array([-1, 0, 1]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_raises_bad_scale(self):
        with pytest.raises(ValueError):
            dcchy(0, scale=0)
