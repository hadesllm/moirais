"""Tests for moirais.fn.zsidp -- IDW power parameter optimization"""

import numpy as np
import pytest

from moirais.fn.zsidp import idw_power


class TestIdwPower:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = idw_power(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = idw_power(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
