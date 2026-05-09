"""Tests for moirais.fn.svnbg -- Nash bargaining in spatial game"""

import numpy as np
import pytest

from moirais.fn.svnbg import nash_bargain_sp


class TestNashBargainSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = nash_bargain_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = nash_bargain_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
