"""Tests for morie.fn.chng — changepoint detection."""
import numpy as np
import pytest
from morie.fn.chng import changepoint_detect


class TestChangepoint:
    def test_detects_shift(self):
        y = np.concatenate([np.zeros(50), np.ones(50) * 5])
        res = changepoint_detect(y)
        assert 40 <= res.extra["changepoint_index"] <= 60
        assert res.extra["detected"]

    def test_no_change(self):
        y = np.ones(100)
        res = changepoint_detect(y)
        assert not res.extra["detected"]
