"""Tests for morie.fn.disp -- Disparity index."""

import pytest
from morie.fn.disp import disparity_index, disp
from morie.fn._containers import ESRes


class TestDisp:
    def test_alias(self):
        assert disp is disparity_index

    def test_basic_ratio(self):
        result = disparity_index(0.4, 0.2)
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(2.0)
        assert result.extra["absolute_diff"] == pytest.approx(0.2)

    def test_parity(self):
        result = disparity_index(0.3, 0.3)
        assert result.estimate == pytest.approx(1.0)
