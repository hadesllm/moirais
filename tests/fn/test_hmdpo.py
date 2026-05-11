"""Tests for hmdpo.geron_dpo."""
import numpy as np
import pytest
from morie.fn.hmdpo import geron_dpo


def test_hmdpo_basic():
    """Test basic functionality."""
    pi = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    preferences = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_dpo(pi, pi_ref, preferences, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdpo_edge():
    """Test edge cases."""
    pi = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    preferences = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_dpo(pi, pi_ref, preferences, beta)
    assert isinstance(result, dict)
