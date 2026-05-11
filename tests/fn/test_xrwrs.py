"""Tests for morie.fn.xrwrs -- Row-standardize weights"""

import numpy as np
import pytest

from morie.fn.xrwrs import w_row_std


class TestWRowStd:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_row_std(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_row_std(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
