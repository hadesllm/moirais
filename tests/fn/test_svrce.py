"""Tests for moirais.fn.svrce -- Roll call classification error"""

import numpy as np
import pytest

from moirais.fn.svrce import roll_call_error


class TestRollCallError:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_call_error(data)
        assert result.value is not None

    def test_output_type(self):
        result = roll_call_error(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
