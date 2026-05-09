"""Tests for moirais.fn.xrwqn -- Queen contiguity weights"""

import numpy as np
import pytest

from moirais.fn.xrwqn import w_queen


class TestWQueen:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_queen(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_queen(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
