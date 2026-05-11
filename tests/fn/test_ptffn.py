"""Tests for morie.fn.ptffn -- Empty space F-function"""

import numpy as np
import pytest

from morie.fn.ptffn import f_function


class TestFFunction:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = f_function(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = f_function(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
