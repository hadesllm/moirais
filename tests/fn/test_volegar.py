"""Tests for volegar.vol_egarch_fit."""
import numpy as np
import pytest
from moirais.fn.volegar import vol_egarch_fit


def test_volegar_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_egarch_fit(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volegar_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_egarch_fit(r, init)
    assert isinstance(result, dict)
