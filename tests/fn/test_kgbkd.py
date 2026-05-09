"""Tests for moirais.fn.kgbkd -- Block kriging discretization"""

import numpy as np
import pytest

from moirais.fn.kgbkd import bk_discretize


class TestBkDiscretize:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bk_discretize(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = bk_discretize(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
