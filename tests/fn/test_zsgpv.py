"""Tests for moirais.fn.zsgpv -- GP predictive variance"""

import numpy as np
import pytest

from moirais.fn.zsgpv import gp_variance


class TestGpVariance:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gp_variance(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gp_variance(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
