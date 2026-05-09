"""Tests for emaxr.em_step_random_effects."""
import numpy as np
import pytest
from moirais.fn.emaxr import em_step_random_effects


def test_emaxr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_u = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_e = np.random.default_rng(42).normal(0, 1, 100)
    result = em_step_random_effects(y, X, cluster, sigma2_u, sigma2_e)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_emaxr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_u = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_e = np.random.default_rng(42).normal(0, 1, 100)
    result = em_step_random_effects(y, X, cluster, sigma2_u, sigma2_e)
    assert isinstance(result, dict)
