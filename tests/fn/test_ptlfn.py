"""Tests for morie.fn.ptlfn -- L-function (variance-stabilized K)"""

import numpy as np
import pytest

from morie.fn.ptlfn import l_function


class TestLFunction:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = l_function(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = l_function(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
