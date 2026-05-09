"""Tests for moirais.fn.ptdgm -- Diggle-Cressie-Loosmore test"""

import numpy as np
import pytest

from moirais.fn.ptdgm import diggle_test


class TestDiggleTest:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = diggle_test(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = diggle_test(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
