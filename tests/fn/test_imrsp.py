"""Test impulse_response (imrsp)."""
import numpy as np
from morie.fn.imrsp import impulse_response, imrsp
from morie.fn._containers import SignalResult


class TestImpulseResponse:
    def test_basic(self):
        b = [1.0, 0.5]
        a = [1.0]
        result = impulse_response(b, a, N=50)
        assert isinstance(result, SignalResult)
        assert result.name == "impulse_response"
        assert result.n_samples == 50

    def test_fir(self):
        b = [1.0, 0.5, 0.25]
        a = [1.0]
        result = impulse_response(b, a, N=10)
        assert abs(result.filtered[0] - 1.0) < 1e-10
        assert abs(result.filtered[1] - 0.5) < 1e-10
        assert abs(result.filtered[2] - 0.25) < 1e-10

    def test_alias(self):
        assert imrsp is impulse_response
