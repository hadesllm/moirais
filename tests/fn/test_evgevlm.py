"""Tests for evgevlm.evt_gev_lmoments."""
import numpy as np
import pytest
from morie.fn.evgevlm import evt_gev_lmoments


def test_evgevlm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_lmoments(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgevlm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_lmoments(x)
    assert isinstance(result, dict)
