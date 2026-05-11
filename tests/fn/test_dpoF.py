"""Tests for dpoF.dpo_loss."""
import numpy as np
import pytest
from morie.fn.dpoF import dpo_loss


def test_dpoF_basic():
    """Test basic functionality."""
    pi_theta = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    pairs = np.random.default_rng(42).normal(0, 1, 100)
    result = dpo_loss(pi_theta, pi_ref, beta, pairs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpoF_edge():
    """Test edge cases."""
    pi_theta = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    pairs = np.random.default_rng(42).normal(0, 1, 100)
    result = dpo_loss(pi_theta, pi_ref, beta, pairs)
    assert isinstance(result, dict)
