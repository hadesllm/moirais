"""Tests for evbevsim.evt_bv_evd_sim."""
import numpy as np
import pytest
from moirais.fn.evbevsim import evt_bv_evd_sim


def test_evbevsim_basic():
    """Test basic functionality."""
    alpha = 0.05
    n = 100
    result = evt_bv_evd_sim(alpha, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evbevsim_edge():
    """Test edge cases."""
    alpha = 0.05
    n = 100
    result = evt_bv_evd_sim(alpha, n)
    assert isinstance(result, dict)
