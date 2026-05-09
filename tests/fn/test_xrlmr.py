"""Tests for moirais.fn.xrlmr -- Robust LM test for lag"""

import numpy as np
import pytest

from moirais.fn.xrlmr import lm_robust_lag


class TestLmRobustLag:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lm_robust_lag(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = lm_robust_lag(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
