"""Tests for evgevm.evt_gev_mle."""
import numpy as np
import pytest
from moirais.fn.evgevm import evt_gev_mle


def test_evgevm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_mle(x, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgevm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_mle(x, init)
    assert isinstance(result, dict)
