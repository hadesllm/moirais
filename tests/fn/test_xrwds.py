"""Tests for morie.fn.xrwds -- Distance-band weights"""

import numpy as np
import pytest

from morie.fn.xrwds import w_distance


class TestWDistance:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_distance(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_distance(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
