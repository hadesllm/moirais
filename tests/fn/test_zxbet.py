"""Tests for morie.fn.zxbet -- Betti numbers computation"""

import numpy as np
import pytest

from morie.fn.zxbet import betti_numbers


class TestBettiNumbers:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = betti_numbers(data)
        assert result.value is not None

    def test_output_type(self):
        result = betti_numbers(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
