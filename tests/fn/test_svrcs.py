"""Tests for moirais.fn.svrcs -- Roll call simulation"""

import numpy as np
import pytest

from moirais.fn.svrcs import roll_call_sim


class TestRollCallSim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_call_sim(data)
        assert result.value is not None

    def test_output_type(self):
        result = roll_call_sim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
