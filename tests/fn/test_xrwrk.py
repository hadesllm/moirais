"""Tests for morie.fn.xrwrk -- Rook contiguity weights"""

import numpy as np
import pytest

from morie.fn.xrwrk import w_rook


class TestWRook:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_rook(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_rook(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
