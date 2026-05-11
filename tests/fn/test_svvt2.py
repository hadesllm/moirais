"""Tests for morie.fn.svvt2 -- 2D vote trading equilibrium"""

import numpy as np
import pytest

from morie.fn.svvt2 import vote_trade_2d


class TestVoteTrade2d:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = vote_trade_2d(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = vote_trade_2d(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
