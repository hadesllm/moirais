"""Tests for moirais.fn.xrgwk -- GWR kernel function"""

import numpy as np
import pytest

from moirais.fn.xrgwk import gwr_kernel


class TestGwrKernel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gwr_kernel(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gwr_kernel(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
