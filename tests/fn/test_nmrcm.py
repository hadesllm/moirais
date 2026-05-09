"""Tests for moirais.fn.nmrcm -- Roll call matrix construction"""

import numpy as np
import pytest

from moirais.fn.nmrcm import roll_call_matrix


class TestRollCallMatrix:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_call_matrix(data)
        assert result.value is not None

    def test_output_type(self):
        result = roll_call_matrix(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
