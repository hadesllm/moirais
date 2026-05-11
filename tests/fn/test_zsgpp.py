"""Tests for morie.fn.zsgpp -- GP prediction"""

import numpy as np
import pytest

from morie.fn.zsgpp import gp_predict


class TestGpPredict:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gp_predict(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gp_predict(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
