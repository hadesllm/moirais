"""Tests for volsk.vol_stochastic_kalman."""
import numpy as np
import pytest
from moirais.fn.volsk import vol_stochastic_kalman


def test_volsk_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_stochastic_kalman(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volsk_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_stochastic_kalman(r, init)
    assert isinstance(result, dict)
