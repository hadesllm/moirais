"""Tests for morie.fn.msnm2 -- Nonmetric MDS 2D"""

import numpy as np
import pytest

from morie.fn.msnm2 import nonmetric_2d


class TestNonmetric2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = nonmetric_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = nonmetric_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
