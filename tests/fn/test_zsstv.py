"""Tests for moirais.fn.zsstv -- Space-time kriging variance"""

import numpy as np
import pytest

from moirais.fn.zsstv import st_kriging_var


class TestStKrigingVar:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = st_kriging_var(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = st_kriging_var(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
