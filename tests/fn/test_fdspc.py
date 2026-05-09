"""Test fractal_dim_from_psd."""
import numpy as np
from moirais.fn.fdspc import fractal_dim_from_psd, fdspc
from moirais.fn._containers import DescriptiveResult


class TestFractalDimFromPSD:
    def test_basic(self):
        freqs = np.linspace(0.1, 50, 100)
        psd = 1.0 / freqs ** 2
        result = fractal_dim_from_psd(psd, freqs)
        assert isinstance(result, DescriptiveResult)

    def test_value_is_float(self):
        freqs = np.linspace(0.1, 50, 100)
        psd = 1.0 / freqs ** 2
        result = fractal_dim_from_psd(psd, freqs)
        assert isinstance(result.value, float)

    def test_white_noise_fd(self):
        freqs = np.linspace(0.1, 50, 100)
        psd = np.ones(100)
        result = fractal_dim_from_psd(psd, freqs)
        assert abs(result.value - 2.5) < 0.5

    def test_name(self):
        freqs = np.linspace(0.1, 50, 100)
        psd = 1.0 / freqs ** 2
        result = fractal_dim_from_psd(psd, freqs)
        assert result.name == "fractal_dim_psd"

    def test_alias(self):
        assert fdspc is fractal_dim_from_psd
