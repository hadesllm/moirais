"""Tests for moirais.fn.zethl -- Spatial Theil decomposition"""

import numpy as np
import pytest

from moirais.fn.zethl import theil_spatial


class TestTheilSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = theil_spatial(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = theil_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
