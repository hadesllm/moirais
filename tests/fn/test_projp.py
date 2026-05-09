"""Test projection_pursuit (projp)."""
import numpy as np
from moirais.fn.projp import projection_pursuit, projp
from moirais.fn._containers import DescriptiveResult


class TestProjp:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        result = projection_pursuit(X, n_components=2, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "projection_pursuit"
        assert result.value == 2

    def test_projections_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        r = projection_pursuit(X, n_components=3, seed=0)
        assert r.extra["projections"].shape == (100, 3)
        assert len(r.extra["negentropies"]) == 3

    def test_alias(self):
        assert projp is projection_pursuit
