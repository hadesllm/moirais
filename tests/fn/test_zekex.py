"""Tests for moirais.fn.zekex -- Kernel density exposure"""

import numpy as np
import pytest

from moirais.fn.zekex import kernel_exposure


class TestKernelExposure:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = kernel_exposure(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = kernel_exposure(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
