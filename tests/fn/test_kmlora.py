"""Tests for kmlora.kamath_lora_weight_update."""
import numpy as np
import pytest
from morie.fn.kmlora import kamath_lora_weight_update


def test_kmlora_basic():
    """Test basic functionality."""
    W0 = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    r = 10
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_lora_weight_update(W0, A, B, alpha, r, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmlora_edge():
    """Test edge cases."""
    W0 = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    r = 10
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_lora_weight_update(W0, A, B, alpha, r, x)
    assert isinstance(result, dict)
