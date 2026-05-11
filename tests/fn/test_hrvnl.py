"""Tests for hrvnl — HRV nonlinear metrics."""
import numpy as np
from morie.fn.hrvnl import hrv_nonlinear
from morie.fn._containers import DescriptiveResult


def test_hrvnl_basic(rng):
    rr = 800 + rng.standard_normal(100) * 50
    result = hrv_nonlinear(rr)
    assert isinstance(result, DescriptiveResult)
    assert "sd1" in result.extra
    assert "sd2" in result.extra


def test_hrvnl_sd1_less_than_sd2(rng):
    rr = 800 + rng.standard_normal(200) * 50
    result = hrv_nonlinear(rr)
    assert result.extra["sd1"] > 0
    assert result.extra["sd2"] > 0
