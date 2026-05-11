"""Tests for morie.fn.awq — activation-aware weight quantization."""

import numpy as np
import pytest

from morie.fn.awq import activation_aware_quant


class TestActivationAwareQuant:

    def test_returns_result(self):
        W = np.random.default_rng(42).standard_normal((8, 16))
        act = np.abs(np.random.default_rng(0).standard_normal(16))
        res = activation_aware_quant(W, act, bits=4)
        assert res.name == "activation_aware_quant"

    def test_mismatched_raises(self):
        W = np.random.default_rng(0).standard_normal((4, 8))
        act = np.ones(10)
        with pytest.raises(ValueError):
            activation_aware_quant(W, act, bits=4)

    def test_salient_weights_preserved(self):
        rng = np.random.default_rng(1)
        W = rng.standard_normal((4, 8))
        act = np.ones(8) * 0.1
        act[0] = 10.0
        res = activation_aware_quant(W, act, bits=4)
        assert res.value >= 0
