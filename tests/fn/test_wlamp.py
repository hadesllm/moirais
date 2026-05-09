"""Test willison_amp."""
import numpy as np
from moirais.fn.wlamp import willison_amp, alias
from moirais.fn._containers import DescriptiveResult


class TestWillisonAmp:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = willison_amp(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_non_negative(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = willison_amp(x)
        assert isinstance(result.value, int)
        assert result.value >= 0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = willison_amp(x)
        assert result.name == "willison_amplitude"

    def test_alias(self):
        assert alias is willison_amp
