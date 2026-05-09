"""Tests for moirais.fn.xrgwr -- GWR basic estimation"""

import numpy as np
import pytest

from moirais.fn.xrgwr import gwr_basic


class TestGwrBasic:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gwr_basic(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gwr_basic(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
