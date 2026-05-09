"""Tests for moirais.fn.ptgfn -- Nearest-neighbor G-function"""

import numpy as np
import pytest

from moirais.fn.ptgfn import g_function


class TestGFunction:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = g_function(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = g_function(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
