"""Tests for moirais.fn.xrlmb -- Robust LM test for error"""

import numpy as np
import pytest

from moirais.fn.xrlmb import lm_robust_error


class TestLmRobustError:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lm_robust_error(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = lm_robust_error(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
