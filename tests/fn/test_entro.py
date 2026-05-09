"""Tests for moirais.fn.entro — Shannon entropy."""
import numpy as np
import pytest
from moirais.fn.entro import entropy


class TestEntropy:
    def test_uniform_high_entropy(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 100, 1000)
        res = entropy(x)
        assert res.extra["entropy"] > 1.0

    def test_constant_low_entropy(self):
        x = np.ones(100)
        res = entropy(x, method="plugin")
        assert res.extra["entropy"] < 1e-10

    def test_base_parameter(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        res_bits = entropy(x, base=2)
        res_nats = entropy(x, base=np.e)
        assert res_bits.extra["entropy"] > 0
        assert res_nats.extra["entropy"] > 0
        ratio = res_bits.extra["entropy"] / max(res_nats.extra["entropy"], 1e-15)
        np.testing.assert_allclose(ratio, 1.0 / np.log(2), rtol=0.1)
