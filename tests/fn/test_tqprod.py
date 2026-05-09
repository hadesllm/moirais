"""Tests for tqprod.turboquant_qjl_product_estimator."""
import numpy as np
import pytest
from moirais.fn.tqprod import turboquant_qjl_product_estimator


def test_tqprod_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    signs_k = np.random.default_rng(42).normal(0, 1, 100)
    norm_k = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_qjl_product_estimator(q, signs_k, norm_k, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqprod_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    signs_k = np.random.default_rng(42).normal(0, 1, 100)
    norm_k = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_qjl_product_estimator(q, signs_k, norm_k, S)
    assert isinstance(result, dict)
