"""Tests for morie.fn.grnln -- Green's function convolution."""

import numpy as np
from morie.fn.grnln import greens_convolve, grnln
from morie.fn._containers import DescriptiveResult


class TestGrnln:
    def test_alias(self):
        assert grnln is greens_convolve

    def test_smooths_signal(self):
        rng = np.random.default_rng(42)
        sig = np.sin(np.linspace(0, 4 * np.pi, 200)) + rng.normal(0, 0.5, 200)
        result = greens_convolve(sig, kernel="gaussian", sigma=3.0)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) == 200

    def test_kernels(self):
        sig = np.ones(50)
        for k in ["gaussian", "exponential", "lorentzian", "heat"]:
            result = greens_convolve(sig, kernel=k)
            assert len(result.value) == 50
