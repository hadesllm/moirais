"""Test dft_compute (dft)."""
import numpy as np
from moirais.fn.dft import dft_compute, dft
from moirais.fn._containers import DescriptiveResult


class TestDft:
    def test_basic(self):
        x = np.array([1.0, 0.0, -1.0, 0.0])
        result = dft_compute(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "dft_compute"

    def test_dc_component(self):
        x = np.ones(8)
        result = dft_compute(x)
        assert np.isclose(result.extra["spectrum"][0], 8.0)

    def test_matches_fft(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(16)
        result = dft_compute(x)
        expected = np.fft.fft(x)
        assert np.allclose(result.extra["spectrum"], expected, atol=1e-10)

    def test_alias(self):
        assert dft is dft_compute
