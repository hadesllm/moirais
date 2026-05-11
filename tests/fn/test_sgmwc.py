"""Test sgmwc."""
import numpy as np
from morie.fn.sgmwc import moving_window_cov


def test_sgmwc_basic():
    r = moving_window_cov()
    assert r.statistic is not None


def test_sgmwc_name():
    r = moving_window_cov()
    assert r.name
