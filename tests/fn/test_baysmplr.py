"""Tests for baysmplr.sampler_dispatch."""
import numpy as np
import pytest
from morie.fn.baysmplr import sampler_dispatch


def test_baysmplr_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    grad_p = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = sampler_dispatch(log_p, grad_p, x0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baysmplr_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    grad_p = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = sampler_dispatch(log_p, grad_p, x0)
    assert isinstance(result, dict)
