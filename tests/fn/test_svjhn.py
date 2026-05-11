"""Tests for morie.fn.svjhn -- Johnston power index"""

import numpy as np
import pytest

from morie.fn.svjhn import johnston_power


class TestJohnstonPower:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = johnston_power(data)
        assert result.value is not None

    def test_output_type(self):
        result = johnston_power(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
