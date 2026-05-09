"""Tests for jkrand.jackknife_repl."""
import numpy as np
import pytest
from moirais.fn.jkrand import jackknife_repl


def test_jkrand_basic():
    """Test basic functionality."""
    theta_replicates = np.random.default_rng(42).normal(0, 1, 100)
    result = jackknife_repl(theta_replicates)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jkrand_edge():
    """Test edge cases."""
    theta_replicates = np.random.default_rng(42).normal(0, 1, 100)
    result = jackknife_repl(theta_replicates)
    assert isinstance(result, dict)
