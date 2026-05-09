"""Tests for moirais.fn.svdm2 -- Second dimensionality test"""

import numpy as np
import pytest

from moirais.fn.svdm2 import dim_test_2


class TestDimTest2:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dim_test_2(data)
        assert result.value is not None

    def test_output_type(self):
        result = dim_test_2(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
