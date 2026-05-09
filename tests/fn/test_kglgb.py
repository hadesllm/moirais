"""Tests for moirais.fn.kglgb -- Lognormal kriging back-transform"""

import numpy as np
import pytest

from moirais.fn.kglgb import lk_backtransform


class TestLkBacktransform:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lk_backtransform(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = lk_backtransform(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
