"""Tests for moirais.fn.kern -- Gaussian KDE."""

import numpy as np
from moirais.fn.kern import kde, kern
from moirais.fn._containers import DescriptiveResult


class TestKern:
    def test_alias(self):
        assert kern is kde

    def test_peak_near_mean(self):
        rng = np.random.default_rng(42)
        x = rng.normal(5.0, 1.0, 200)
        result = kde(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 5.0) < 1.0

    def test_density_shape(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        result = kde(x, n_grid=100)
        assert len(result.extra["grid"]) == 100
        assert len(result.extra["density"]) == 100
