"""Test imf_criteria (imfcr)."""
import numpy as np
from moirais.fn.imfcr import imf_criteria, imfcr
from moirais.fn._containers import DescriptiveResult


class TestImfcr:
    def test_basic(self):
        t = np.linspace(0, 1, 128)
        x = np.sin(2 * np.pi * 5 * t)
        result = imf_criteria(x, x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "imf_criteria"

    def test_pure_sine_passes(self):
        t = np.linspace(0, 2, 256)
        imf = np.sin(2 * np.pi * 3 * t)
        r = imf_criteria(imf, imf + 0.1 * np.sin(2 * np.pi * 20 * t))
        assert "s_number" in r.extra
        assert "energy_ratio" in r.extra

    def test_alias(self):
        assert imfcr is imf_criteria
