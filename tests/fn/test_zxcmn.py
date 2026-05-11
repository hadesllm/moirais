"""Tests for morie.fn.zxcmn -- Spatial circular mean"""

import numpy as np
import pytest

from morie.fn.zxcmn import circular_mean_sp


class TestCircularMeanSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = circular_mean_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = circular_mean_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
