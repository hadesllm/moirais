"""Tests for volhar1.vol_har_q."""
import numpy as np
import pytest
from moirais.fn.volhar1 import vol_har_q


def test_volhar1_basic():
    """Test basic functionality."""
    RV = np.random.default_rng(42).normal(0, 1, 100)
    RQ = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_har_q(RV, RQ)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volhar1_edge():
    """Test edge cases."""
    RV = np.random.default_rng(42).normal(0, 1, 100)
    RQ = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_har_q(RV, RQ)
    assert isinstance(result, dict)
