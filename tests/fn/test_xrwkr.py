"""Tests for moirais.fn.xrwkr -- Kernel weights"""

import numpy as np
import pytest

from moirais.fn.xrwkr import w_kernel


class TestWKernel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_kernel(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_kernel(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
