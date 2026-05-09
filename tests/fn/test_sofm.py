"""Test self_org_map (sofm)."""
import numpy as np
from moirais.fn.sofm import self_org_map, sofm
from moirais.fn._containers import DescriptiveResult


class TestSofm:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4))
        result = self_org_map(X, grid_size=(3, 3), n_iter=100)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "self_org_map"
        assert result.extra["weights"].shape == (3, 3, 4)

    def test_bmu_indices(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        result = self_org_map(X, grid_size=(2, 2), n_iter=50)
        assert len(result.extra["bmu_indices"]) == 30

    def test_alias(self):
        assert sofm is self_org_map
