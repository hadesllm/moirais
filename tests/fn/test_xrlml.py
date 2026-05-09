"""Tests for moirais.fn.xrlml -- LM test for spatial lag"""

import numpy as np
import pytest

from moirais.fn.xrlml import lm_lag


class TestLmLag:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lm_lag(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = lm_lag(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
