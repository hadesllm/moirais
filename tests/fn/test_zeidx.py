"""Tests for morie.fn.zeidx -- IDW exposure interpolation"""

import numpy as np
import pytest

from morie.fn.zeidx import idw_exposure


class TestIdwExposure:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = idw_exposure(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = idw_exposure(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
