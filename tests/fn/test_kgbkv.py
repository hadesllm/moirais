"""Tests for morie.fn.kgbkv -- Block kriging variance"""

import numpy as np
import pytest

from morie.fn.kgbkv import bk_variance


class TestBkVariance:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bk_variance(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = bk_variance(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
