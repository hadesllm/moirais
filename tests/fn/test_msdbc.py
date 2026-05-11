"""Tests for morie.fn.msdbc -- Double centering matrix B"""

import numpy as np
import pytest

from morie.fn.msdbc import double_center


class TestDoubleCenter:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = double_center(data)
        assert result.value is not None

    def test_output_type(self):
        result = double_center(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
