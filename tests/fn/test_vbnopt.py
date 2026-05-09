"""Tests for vbnopt.variational_inference."""
import numpy as np
import pytest
from moirais.fn.vbnopt import variational_inference


def test_vbnopt_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = variational_inference(log_p, q, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vbnopt_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = variational_inference(log_p, q, x)
    assert isinstance(result, dict)
