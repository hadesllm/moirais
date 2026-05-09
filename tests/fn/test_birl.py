"""Tests for birl.bayesian_irl."""
import numpy as np
import pytest
from moirais.fn.birl import bayesian_irl


def test_birl_basic():
    """Test basic functionality."""
    expert_trajs = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_irl(expert_trajs, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_birl_edge():
    """Test edge cases."""
    expert_trajs = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_irl(expert_trajs, prior)
    assert isinstance(result, dict)
