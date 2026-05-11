"""Test add_noise (addns)."""
import numpy as np
from morie.fn.addns import add_noise, addns
from morie.fn._containers import DescriptiveResult


class TestAddNoise:
    def test_length(self):
        s = np.sin(np.linspace(0, 2 * np.pi, 100))
        result = add_noise(s, snr_db=20.0, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) == 100

    def test_high_snr_close(self):
        s = np.ones(100)
        result = add_noise(s, snr_db=60.0, seed=42)
        assert np.allclose(result.value, s, atol=0.1)

    def test_alias(self):
        assert addns is add_noise
