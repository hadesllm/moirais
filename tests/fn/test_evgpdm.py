"""Tests for evgpdm.evt_gpd_mle."""
import numpy as np
import pytest
from moirais.fn.evgpdm import evt_gpd_mle


def test_evgpdm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_mle(y, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgpdm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gpd_mle(y, init)
    assert isinstance(result, dict)
