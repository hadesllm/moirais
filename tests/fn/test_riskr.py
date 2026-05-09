"""Tests for riskr.risch_integration."""
import numpy as np
import pytest
from moirais.fn.riskr import risch_integration


def test_riskr_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = risch_integration(expr, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_riskr_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = risch_integration(expr, x)
    assert isinstance(result, dict)
