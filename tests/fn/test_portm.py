"""Tests for moirais.fn.portm -- Portmanteau (Ljung-Box) test."""

import numpy as np
import pytest
from moirais.fn.portm import portmanteau_test


class TestPortmanteau:
    def test_iid_not_rejected(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        r = portmanteau_test(x, lags=10)
        assert not r["reject"]
        assert r["p_value"] > 0.05

    def test_ar1_rejected(self):
        rng = np.random.default_rng(42)
        n = 300
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + rng.standard_normal()
        r = portmanteau_test(x, lags=10)
        assert r["reject"]

    def test_box_pierce_variant(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        r = portmanteau_test(x, method="box-pierce")
        assert r["method"] == "box-pierce"
        assert r["Q"] > 0

    def test_too_few_obs(self):
        with pytest.raises(ValueError, match="at least"):
            portmanteau_test(np.array([1, 2, 3]), lags=10)

    def test_autocorrelations_length(self):
        rng = np.random.default_rng(42)
        r = portmanteau_test(rng.standard_normal(100), lags=5)
        assert len(r["autocorrelations"]) == 5
