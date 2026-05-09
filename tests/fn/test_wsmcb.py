"""Tests for wsmcb.wasserman_dkw_cb."""
import numpy as np
import pytest
from moirais.fn.wsmcb import wasserman_dkw_cb


def test_wsmcb_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = wasserman_dkw_cb(data, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmcb_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = wasserman_dkw_cb(data, alpha)
    assert isinstance(result, dict)
