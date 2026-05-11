"""Test idft_compute (idft)."""
import numpy as np
from morie.fn.idft import idft_compute, idft
from morie.fn._containers import DescriptiveResult


class TestIdft:
    def test_basic(self):
        X = np.array([4.0+0j, 0.0+0j, 0.0+0j, 0.0+0j])
        result = idft_compute(X)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "idft_compute"

    def test_roundtrip(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.fft.fft(x)
        result = idft_compute(X)
        assert np.allclose(np.real(result.extra["signal"]), x, atol=1e-10)

    def test_alias(self):
        assert idft is idft_compute
