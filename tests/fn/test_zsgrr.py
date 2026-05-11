"""Tests for morie.fn.zsgrr -- Grid resampling"""

import numpy as np
import pytest

from morie.fn.zsgrr import grid_resample


class TestGridResample:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = grid_resample(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = grid_resample(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
