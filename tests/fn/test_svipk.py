"""Tests for morie.fn.svipk -- Kernel smoothed ideal point"""

import numpy as np
import pytest

from morie.fn.svipk import ideal_point_kernel


class TestIdealPointKernel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ideal_point_kernel(data)
        assert result.value is not None

    def test_output_type(self):
        result = ideal_point_kernel(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
