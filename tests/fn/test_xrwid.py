"""Tests for morie.fn.xrwid -- Inverse distance weights matrix"""

import numpy as np
import pytest

from morie.fn.xrwid import w_inverse_dist


class TestWInverseDist:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_inverse_dist(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_inverse_dist(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
