"""Tests for moirais.fn.ptjfn -- J-function (ratio F/G)"""

import numpy as np
import pytest

from moirais.fn.ptjfn import j_function


class TestJFunction:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = j_function(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = j_function(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
