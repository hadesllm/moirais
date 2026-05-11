"""Tests for irtnm -- Nominal Response Model."""
import numpy as np
from morie.fn.irtnm import irt_nominal
from morie.fn._containers import IRTResult


class TestIrtNominal:
    def test_basic_fit(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 4, size=(100, 5))
        result = irt_nominal(data, n_categories=4)
        assert isinstance(result, IRTResult)
        assert result.model == "NRM"
        assert len(result.item_params) == 5

    def test_theta_length(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 3, size=(80, 4))
        result = irt_nominal(data, n_categories=3)
        assert len(result.theta) == 80
