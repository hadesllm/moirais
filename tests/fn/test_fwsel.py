"""Test forward_select (fwsel)."""
import numpy as np
from moirais.fn.fwsel import forward_select, fwsel
from moirais.fn._containers import DescriptiveResult


class TestFwsel:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 5))
        y = (X[:, 0] > 0).astype(int)
        result = forward_select(X, y, max_features=3, cv=3)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "forward_select"
        assert result.extra["n_selected"] <= 3

    def test_informative_first(self):
        rng = np.random.default_rng(42)
        X = np.column_stack([rng.standard_normal(60), rng.standard_normal((60, 4)) * 0.01])
        y = (X[:, 0] > 0).astype(int)
        result = forward_select(X, y, max_features=1, cv=3)
        assert result.extra["selected_features"][0] == 0

    def test_alias(self):
        assert fwsel is forward_select
