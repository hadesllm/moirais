"""Tests for moirais.fn.xrwcn -- Weights connectivity check"""

import numpy as np
import pytest

from moirais.fn.xrwcn import w_connected


class TestWConnected:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_connected(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_connected(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
