"""Tests for irtrs -- Rating Scale Model."""
import numpy as np
from moirais.fn.irtrs import irt_rsm
from moirais.fn._containers import IRTResult


class TestIrtRsm:
    def test_basic_fit(self):
        rng = np.random.default_rng(42)
        data = np.clip(np.round(rng.normal(3, 1, (100, 5))).astype(int), 1, 5)
        result = irt_rsm(data)
        assert isinstance(result, IRTResult)
        assert result.model == "RSM"

    def test_params_have_tau(self):
        rng = np.random.default_rng(42)
        data = np.clip(np.round(rng.normal(3, 1, (80, 4))).astype(int), 1, 5)
        result = irt_rsm(data)
        for p in result.item_params.values():
            assert "tau" in p
            assert "delta" in p
