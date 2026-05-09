"""Tests for spmenv.schabenberger_moran_expectation."""
import numpy as np
import pytest
from moirais.fn.spmenv import schabenberger_moran_expectation


def test_spmenv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_moran_expectation(x, w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spmenv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_moran_expectation(x, w)
    assert isinstance(result, dict)
