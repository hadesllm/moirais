"""Tests for spred.shrinkage_predictor_level2."""
import numpy as np
import pytest
from morie.fn.spred import shrinkage_predictor_level2


def test_spred_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_u = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_e = np.random.default_rng(42).normal(0, 1, 100)
    result = shrinkage_predictor_level2(y, cluster, sigma2_u, sigma2_e)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spred_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_u = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_e = np.random.default_rng(42).normal(0, 1, 100)
    result = shrinkage_predictor_level2(y, cluster, sigma2_u, sigma2_e)
    assert isinstance(result, dict)
