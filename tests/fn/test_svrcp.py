"""Tests for morie.fn.svrcp -- Roll call vote probability model"""

import numpy as np
import pytest

from morie.fn.svrcp import roll_call_prob


class TestRollCallProb:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_call_prob(data)
        assert result.value is not None

    def test_output_type(self):
        result = roll_call_prob(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
