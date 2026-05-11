"""Tests for morie.fn.zemch -- Maternal-child health mapping"""

import numpy as np
import pytest

from morie.fn.zemch import maternal_child_map


class TestMaternalChildMap:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = maternal_child_map(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = maternal_child_map(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
