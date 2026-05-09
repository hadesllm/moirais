"""Tests for drlp1.dr_lp_did."""
import numpy as np
import pytest
from moirais.fn.drlp1 import dr_lp_did


def test_drlp1_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_lp_did(y, D, unit, time, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drlp1_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_lp_did(y, D, unit, time, horizon)
    assert isinstance(result, dict)
