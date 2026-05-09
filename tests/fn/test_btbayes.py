"""Tests for btbayes.boot_bayesian."""
import numpy as np
import pytest
from moirais.fn.btbayes import boot_bayesian


def test_btbayes_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_bayesian(x, stat, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btbayes_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_bayesian(x, stat, B)
    assert isinstance(result, dict)
