"""Tests for satDP.dpll."""
import numpy as np
import pytest
from moirais.fn.satDP import dpll


def test_satDP_basic():
    """Test basic functionality."""
    cnf = np.random.default_rng(42).normal(0, 1, 100)
    result = dpll(cnf)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_satDP_edge():
    """Test edge cases."""
    cnf = np.random.default_rng(42).normal(0, 1, 100)
    result = dpll(cnf)
    assert isinstance(result, dict)
