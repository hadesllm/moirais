"""Tests for moirais.fn.pt2nd -- Second-order point pattern stats"""

import numpy as np
import pytest

from moirais.fn.pt2nd import second_order_pp


class TestSecondOrderPp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = second_order_pp(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = second_order_pp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
