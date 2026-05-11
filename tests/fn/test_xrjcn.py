"""Tests for morie.fn.xrjcn -- Join count statistic"""

import numpy as np
import pytest

from morie.fn.xrjcn import join_count


class TestJoinCount:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = join_count(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = join_count(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
