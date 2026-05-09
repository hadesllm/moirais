"""Tests for moirais.fn.anmls -- multi-view CCA."""

import numpy as np
from moirais.fn.anmls import multiview_cca, anmls
from moirais.fn._containers import DescriptiveResult


class TestAnmls:
    def test_alias(self):
        assert anmls is multiview_cca

    def test_correlated_views(self):
        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, (50, 1))
        X1 = z @ rng.normal(0, 1, (1, 3)) + rng.normal(0, 0.1, (50, 3))
        X2 = z @ rng.normal(0, 1, (1, 4)) + rng.normal(0, 0.1, (50, 4))
        result = multiview_cca(X1, X2, n_components=1)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0.5

    def test_independent(self):
        rng = np.random.default_rng(42)
        X1 = rng.normal(0, 1, (100, 3))
        X2 = rng.normal(0, 1, (100, 3))
        result = multiview_cca(X1, X2)
        assert result.value < 0.5
