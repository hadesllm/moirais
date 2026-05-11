"""Test bilinear_transform (bltrf)."""
import numpy as np
from morie.fn.bltrf import bilinear_transform, bltrf
from morie.fn._containers import DescriptiveResult


class TestBilinearTransform:
    def test_basic(self):
        b_s = [1.0]
        a_s = [1.0, 1.0]
        result = bilinear_transform(b_s, a_s, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "bilinear_transform"

    def test_coefficients(self):
        b_s = [1.0]
        a_s = [1.0, 1.0]
        result = bilinear_transform(b_s, a_s, fs=100.0)
        assert "b_digital" in result.extra
        assert "a_digital" in result.extra

    def test_alias(self):
        assert bltrf is bilinear_transform
