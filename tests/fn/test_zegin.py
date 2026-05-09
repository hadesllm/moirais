"""Tests for moirais.fn.zegin -- Spatial Gini coefficient"""

import numpy as np
import pytest

from moirais.fn.zegin import gini_spatial


class TestGiniSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gini_spatial(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gini_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
