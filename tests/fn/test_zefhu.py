"""Tests for moirais.fn.zefhu -- Unit-level Fay-Herriot"""

import numpy as np
import pytest

from moirais.fn.zefhu import fay_herriot_unit


class TestFayHerriotUnit:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fay_herriot_unit(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = fay_herriot_unit(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
