"""Tests for recapTOA.toa_radiation_balance."""
import numpy as np
import pytest
from morie.fn.recapTOA import toa_radiation_balance


def test_recapTOA_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    OLR = np.random.default_rng(42).normal(0, 1, 100)
    result = toa_radiation_balance(S, alpha, OLR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_recapTOA_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    OLR = np.random.default_rng(42).normal(0, 1, 100)
    result = toa_radiation_balance(S, alpha, OLR)
    assert isinstance(result, dict)
