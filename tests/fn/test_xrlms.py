"""Tests for moirais.fn.xrlms -- LM test SARMA"""

import numpy as np
import pytest

from moirais.fn.xrlms import lm_sarma


class TestLmSarma:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lm_sarma(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = lm_sarma(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
