"""Tests for moirais.fn.zxfkr -- Functional kriging"""

import numpy as np
import pytest

from moirais.fn.zxfkr import func_kriging


class TestFuncKriging:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = func_kriging(vals, x)
        assert result.value is not None

    def test_output_type(self):
        result = func_kriging(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "value")
