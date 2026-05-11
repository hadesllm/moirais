"""Tests for morie.fn.svwt2 -- Wittman model in 2D space"""

import numpy as np
import pytest

from morie.fn.svwt2 import wittman_2d


class TestWittman2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wittman_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = wittman_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
