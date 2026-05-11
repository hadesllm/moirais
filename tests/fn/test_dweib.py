"""Tests for morie.fn.dweib — Weibull PDF."""

import numpy as np
import pytest

from morie.fn.dweib import dweib


class TestDweib:
    """Tests for dweib()."""

    def test_exponential_case(self):
        """dweib(1, shape=1, scale=1) = e^{-1} ~ 0.3679."""
        assert dweib(1, shape=1, scale=1) == pytest.approx(0.3679, abs=1e-3)

    def test_at_zero(self):
        """dweib(0, shape=1, scale=1) = 1.0 (exponential at origin)."""
        assert dweib(0, shape=1, scale=1) == pytest.approx(1.0, abs=1e-6)

    def test_type(self):
        """Scalar input returns float."""
        result = dweib(2.0, shape=2, scale=1)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_shape(self):
        """Should reject shape <= 0."""
        with pytest.raises(ValueError):
            dweib(1, shape=0)
