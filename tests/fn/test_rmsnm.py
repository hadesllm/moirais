"""Test rms_norm."""
import numpy as np
from moirais.fn.rmsnm import rms_norm, rmsnm
from moirais.fn._containers import DescriptiveResult


class TestRmsNorm:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = rms_norm(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "rms_norm"

    def test_unit_rms(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = rms_norm(x)
        out = result.extra["output"]
        rms_out = np.sqrt(np.mean(out**2))
        assert abs(rms_out - 1.0) < 0.01

    def test_with_weight(self):
        x = np.array([1.0, 2.0])
        w = np.array([2.0, 0.5])
        result = rms_norm(x, weight=w)
        assert result.extra["output"] is not None

    def test_alias(self):
        assert rmsnm is rms_norm
