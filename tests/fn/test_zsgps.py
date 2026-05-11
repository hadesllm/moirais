"""Tests for morie.fn.zsgps -- Gaussian process spatial"""

import numpy as np
import pytest

from morie.fn.zsgps import gp_spatial


class TestGpSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gp_spatial(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gp_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
