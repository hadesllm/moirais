"""Tests for morie.fn.svrcl -- Roll call logit model"""

import numpy as np
import pytest

from morie.fn.svrcl import roll_call_logit


class TestRollCallLogit:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roll_call_logit(data)
        assert result.value is not None

    def test_output_type(self):
        result = roll_call_logit(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
