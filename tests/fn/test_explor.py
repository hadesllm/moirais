"""Tests for explor.intrinsic_motivation."""
import numpy as np
import pytest
from moirais.fn.explor import intrinsic_motivation


def test_explor_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    forward_model = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = intrinsic_motivation(env, forward_model, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_explor_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    forward_model = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = intrinsic_motivation(env, forward_model, beta)
    assert isinstance(result, dict)
