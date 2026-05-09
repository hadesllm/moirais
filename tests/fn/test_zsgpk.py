"""Tests for moirais.fn.zsgpk -- GP kernel selection"""

import numpy as np
import pytest

from moirais.fn.zsgpk import gp_kernel


class TestGpKernel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gp_kernel(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gp_kernel(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
