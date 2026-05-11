"""Test normalized_xcorr (nxcor)."""
import numpy as np
from morie.fn.nxcor import normalized_xcorr, nxcor
from morie.fn._containers import DescriptiveResult


class TestNormalizedXcorr:
    def test_self(self):
        x = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = normalized_xcorr(x, x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 1.0) < 1e-10
        assert result.extra["lag"] == 0

    def test_alias(self):
        assert nxcor is normalized_xcorr
