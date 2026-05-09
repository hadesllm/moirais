"""Test dtft_compute (dtft)."""
import numpy as np
from moirais.fn.dtft import dtft_compute, dtft
from moirais.fn._containers import DescriptiveResult


class TestDtft:
    def test_basic(self):
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = dtft_compute(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "dtft_compute"

    def test_dc_at_zero(self):
        x = np.ones(8)
        result = dtft_compute(x, omega=np.array([0.0]))
        assert np.isclose(np.abs(result.extra["spectrum"][0]), 8.0)

    def test_alias(self):
        assert dtft is dtft_compute
