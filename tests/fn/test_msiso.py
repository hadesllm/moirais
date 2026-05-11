"""Tests for morie.fn.msiso -- Isotonic regression for MDS"""

import numpy as np
import pytest

from morie.fn.msiso import isotonic_reg


class TestIsotonicReg:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = isotonic_reg(data)
        assert result.value is not None

    def test_output_type(self):
        result = isotonic_reg(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
