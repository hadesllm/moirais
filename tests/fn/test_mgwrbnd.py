"""Tests for moirais.fn.mgwrbnd."""
import numpy as np
import pytest
from moirais.fn.mgwrbnd import mgwrbnd


class TestMgwrbnd:
    def test_basic(self):
        bw=0.5; se_bw=0.05
        result = mgwrbnd(bw, se_bw)
        assert result is not None

    def test_returns_spatial_result(self):
        bw=0.5; se_bw=0.05
        result = mgwrbnd(bw, se_bw)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        bw=0.5; se_bw=0.05
        result = mgwrbnd(bw, se_bw)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
