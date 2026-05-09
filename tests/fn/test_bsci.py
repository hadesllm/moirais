"""Tests for moirais.fn.bsci -- bootstrap effect-size CI wrapper."""

import numpy as np
import pytest
from moirais.fn.bsci import bootstrap_effect_size_ci


class TestBootstrapEffectSizeCI:
    def test_mean_function(self, rng):
        """Bootstrap CI for np.mean should return ESRes."""
        x = rng.normal(5.0, 1.0, 100)
        result = bootstrap_effect_size_ci(np.mean, x, n_boot=500)
        assert result.measure.startswith("Bootstrap")
        assert result.ci_lower < result.estimate < result.ci_upper

    def test_std_function(self, rng):
        """Bootstrap CI for np.std should work."""
        x = rng.normal(0, 2.0, 200)
        result = bootstrap_effect_size_ci(np.std, x, n_boot=500)
        assert result.estimate > 0
        assert result.ci_lower > 0

    def test_has_se(self, rng):
        """Result should have a standard error."""
        x = rng.standard_normal(50)
        result = bootstrap_effect_size_ci(np.mean, x, n_boot=300)
        assert result.se is not None
        assert result.se > 0
