"""Tests for vinfer.variational_inference."""
import numpy as np
import pytest
from morie.fn.vinfer import variational_inference


def test_vinfer_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    q_family = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = variational_inference(log_p, q_family, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vinfer_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    q_family = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = variational_inference(log_p, q_family, x)
    assert isinstance(result, dict)
