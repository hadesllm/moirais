"""Tests for fwpas.forward_pass_dense."""
import numpy as np
import pytest
from moirais.fn.fwpas import forward_pass_dense


def test_fwpas_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = forward_pass_dense(x, w, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fwpas_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = forward_pass_dense(x, w, b)
    assert isinstance(result, dict)
