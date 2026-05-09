"""Tests for moirais.fn.cvmsv -- Cramer-von Mises test."""

import numpy as np
import pytest
from moirais.fn.cvmsv import cramer_von_mises


class TestCramerVonMises:
    def test_normal_data_not_rejected(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        r = cramer_von_mises(x)
        assert not r["reject"]
        assert r["W2"] > 0

    def test_uniform_data_rejected(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, 200)
        r = cramer_von_mises(x, cdf="norm")
        assert r["reject"]

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 3"):
            cramer_von_mises(np.array([1.0, 2.0]))

    def test_unknown_cdf(self):
        with pytest.raises(ValueError, match="Unknown"):
            cramer_von_mises(np.array([1, 2, 3, 4, 5]), cdf="nonexistent")

    def test_exponential_cdf(self):
        rng = np.random.default_rng(42)
        x = rng.exponential(1.0, 200)
        r = cramer_von_mises(x, cdf="expon")
        assert r["p_value"] > 0
