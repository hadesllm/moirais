"""Tests for morie.fn.svdmt -- Dimensionality test for spatial data"""

import numpy as np
import pytest

from morie.fn.svdmt import dim_test


class TestDimTest:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dim_test(data)
        assert result.value is not None

    def test_output_type(self):
        result = dim_test(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
