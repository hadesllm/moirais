"""Tests for moirais.fn.xrslx -- SLX model OLS estimation"""

import numpy as np
import pytest

from moirais.fn.xrslx import slx_ols


class TestSlxOls:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = slx_ols(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = slx_ols(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
