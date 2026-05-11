"""Tests for morie.fn.svrub -- Rubinstein spatial bargaining"""

import numpy as np
import pytest

from morie.fn.svrub import rubinstein_sp


class TestRubinsteinSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rubinstein_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = rubinstein_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
