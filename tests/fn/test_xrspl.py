"""Tests for moirais.fn.xrspl -- Spatial logit model"""

import numpy as np
import pytest

from moirais.fn.xrspl import spatial_logit


class TestSpatialLogit:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_logit(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_logit(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
