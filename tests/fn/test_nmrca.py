"""Tests for morie.fn.nmrca -- Roll call agreement score"""

import numpy as np
import pytest

from morie.fn.nmrca import roll_call_agree


class TestRollCallAgree:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_call_agree(data)
        assert result.value is not None

    def test_output_type(self):
        result = roll_call_agree(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
