"""Tests for moirais.fn.xrspp -- Spatial probit model"""

import numpy as np
import pytest

from moirais.fn.xrspp import spatial_probit


class TestSpatialProbit:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_probit(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_probit(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
