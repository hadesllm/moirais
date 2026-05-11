"""Tests for morie.fn.svclk -- Coalition kernel set"""

import numpy as np
import pytest

from morie.fn.svclk import coalition_kernel


class TestCoalitionKernel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = coalition_kernel(data)
        assert result.value is not None

    def test_output_type(self):
        result = coalition_kernel(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
