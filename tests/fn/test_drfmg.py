"""Test drift_magnitude (drfmg)."""
import numpy as np

from moirais.fn.drfmg import drift_magnitude, drfmg
from moirais.fn._containers import DescriptiveResult


class TestDriftMagnitude:
    def test_no_drift(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        result = drift_magnitude(x, window=50)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "drift_magnitude"

    def test_linear_drift(self):
        x = np.linspace(0, 10, 1000)
        result = drift_magnitude(x, window=50)
        assert result.value > 5.0

    def test_alias(self):
        assert drfmg is drift_magnitude
