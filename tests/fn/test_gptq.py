"""Tests for moirais.fn.gptq — GPTQ weight quantizer."""

import numpy as np
import pytest

from moirais.fn.gptq import gptq_quantize


class TestGptqQuantize:

    def test_returns_result(self):
        W = np.random.default_rng(42).standard_normal((8, 16))
        res = gptq_quantize(W, bits=4)
        assert res.name == "gptq_quantize"
        assert res.value >= 0

    def test_shape_preserved(self):
        W = np.random.default_rng(0).standard_normal((4, 8))
        res = gptq_quantize(W, bits=4)
        assert res.extra["W_quantized"].shape == (4, 8)

    def test_with_hessian(self):
        W = np.random.default_rng(1).standard_normal((4, 8))
        H = np.eye(8)
        res = gptq_quantize(W, H=H, bits=4)
        assert res.value >= 0

    def test_1d_input(self):
        W = np.random.default_rng(2).standard_normal(16)
        res = gptq_quantize(W, bits=4)
        assert res.extra["shape"] == (1, 16)
