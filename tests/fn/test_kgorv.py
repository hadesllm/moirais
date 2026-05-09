"""Tests for moirais.fn.kgorv -- Ordinary kriging variance"""

import numpy as np
import pytest

from moirais.fn.kgorv import ok_variance


class TestOkVariance:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ok_variance(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = ok_variance(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
