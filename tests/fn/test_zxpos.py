"""Tests for morie.fn.zxpos -- Possibilistic spatial clustering"""

import numpy as np
import pytest

from morie.fn.zxpos import possibilistic_sp


class TestPossibilisticSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = possibilistic_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = possibilistic_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
