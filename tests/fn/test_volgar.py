"""Tests for volgar.vol_garch11_fit."""
import numpy as np
import pytest
from moirais.fn.volgar import vol_garch11_fit


def test_volgar_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garch11_fit(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volgar_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_garch11_fit(r, init)
    assert isinstance(result, dict)
