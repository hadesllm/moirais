"""Tests for smcopt.sequential_mc."""
import numpy as np
import pytest
from moirais.fn.smcopt import sequential_mc


def test_smcopt_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    temperatures = np.random.default_rng(42).normal(0, 1, 100)
    result = sequential_mc(f, x0, temperatures)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smcopt_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    temperatures = np.random.default_rng(42).normal(0, 1, 100)
    result = sequential_mc(f, x0, temperatures)
    assert isinstance(result, dict)
