"""Tests for morie.fn.nmrcn -- Roll call normalization"""

import numpy as np
import pytest

from morie.fn.nmrcn import roll_call_normalize


class TestRollCallNormalize:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_call_normalize(data)
        assert result.value is not None

    def test_output_type(self):
        result = roll_call_normalize(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
