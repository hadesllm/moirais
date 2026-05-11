"""Tests for prdldm.prox_method."""
import numpy as np
import pytest
from morie.fn.prdldm import prox_method


def test_prdldm_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    prox_g = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = prox_method(f, grad_f, prox_g, x0, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prdldm_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    prox_g = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = prox_method(f, grad_f, prox_g, x0, lr)
    assert isinstance(result, dict)
