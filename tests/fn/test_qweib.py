"""Tests for morie.fn.qweib — Weibull quantile function."""

import numpy as np
import pytest

from morie.fn.qweib import qweib


class TestQweib:
    """Tests for qweib()."""

    def test_median_exponential(self):
        """qweib(0.5, shape=1, scale=1) = ln(2) ~ 0.6931."""
        assert qweib(0.5, shape=1, scale=1) == pytest.approx(0.6931, abs=1e-3)

    def test_positive(self):
        """All quantiles are positive."""
        for p in [0.1, 0.5, 0.9]:
            assert qweib(p, shape=2, scale=1) > 0

    def test_type(self):
        """Scalar input returns float."""
        result = qweib(0.5, shape=2, scale=1)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_shape(self):
        """Should reject shape <= 0."""
        with pytest.raises(ValueError):
            qweib(0.5, shape=0)
