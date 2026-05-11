"""Tests for morie.fn.nmrcf -- Roll call filtering"""

import numpy as np
import pytest

from morie.fn.nmrcf import roll_call_filter


class TestRollCallFilter:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_call_filter(data)
        assert result.value is not None

    def test_output_type(self):
        result = roll_call_filter(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
