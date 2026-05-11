"""Tests for morie.fn.msind -- INDSCAL individual differences MDS"""

import numpy as np
import pytest

from morie.fn.msind import indscal


class TestIndscal:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = indscal(data)
        assert result.value is not None

    def test_output_type(self):
        result = indscal(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
