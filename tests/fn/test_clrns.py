"""Test colored_noise_gen (clrns)."""
import numpy as np
from moirais.fn.clrns import colored_noise_gen, clrns
from moirais.fn._containers import DescriptiveResult


class TestColoredNoise:
    def test_length(self):
        result = colored_noise_gen(100, alpha=1.0, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) == 100

    def test_unit_variance(self):
        result = colored_noise_gen(1000, alpha=1.0, seed=42)
        assert abs(np.std(result.value) - 1.0) < 0.1

    def test_alias(self):
        assert clrns is colored_noise_gen
