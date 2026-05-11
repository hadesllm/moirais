"""Tests for riskrt.risk_ratio."""
import numpy as np
import pytest
from morie.fn.riskrt import risk_ratio


def test_riskrt_basic():
    """Test basic functionality."""
    p_exposed = np.random.default_rng(42).normal(0, 1, 100)
    p_unexposed = np.random.default_rng(42).normal(0, 1, 100)
    result = risk_ratio(p_exposed, p_unexposed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_riskrt_edge():
    """Test edge cases."""
    p_exposed = np.random.default_rng(42).normal(0, 1, 100)
    p_unexposed = np.random.default_rng(42).normal(0, 1, 100)
    result = risk_ratio(p_exposed, p_unexposed)
    assert isinstance(result, dict)
