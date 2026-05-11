"""Tests for morie.fn.xrwad -- Adaptive bandwidth weights"""

import numpy as np
import pytest

from morie.fn.xrwad import w_adaptive


class TestWAdaptive:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_adaptive(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_adaptive(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
