"""Tests for morie.fn.pt1st -- First-order point pattern stats"""

import numpy as np
import pytest

from morie.fn.pt1st import first_order_pp


class TestFirstOrderPp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = first_order_pp(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = first_order_pp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
