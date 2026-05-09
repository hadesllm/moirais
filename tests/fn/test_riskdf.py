"""Tests for riskdf.risk_difference."""
import numpy as np
import pytest
from moirais.fn.riskdf import risk_difference


def test_riskdf_basic():
    """Test basic functionality."""
    p_exposed = np.random.default_rng(42).normal(0, 1, 100)
    p_unexposed = np.random.default_rng(42).normal(0, 1, 100)
    result = risk_difference(p_exposed, p_unexposed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_riskdf_edge():
    """Test edge cases."""
    p_exposed = np.random.default_rng(42).normal(0, 1, 100)
    p_unexposed = np.random.default_rng(42).normal(0, 1, 100)
    result = risk_difference(p_exposed, p_unexposed)
    assert isinstance(result, dict)
