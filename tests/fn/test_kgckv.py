"""Tests for morie.fn.kgckv -- Co-kriging variance"""

import numpy as np
import pytest

from morie.fn.kgckv import cok_variance


class TestCokVariance:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cok_variance(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = cok_variance(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
