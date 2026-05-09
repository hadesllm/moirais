"""Tests for hmoptn.geron_optuna."""
import numpy as np
import pytest
from moirais.fn.hmoptn import geron_optuna


def test_hmoptn_basic():
    """Test basic functionality."""
    objective = np.random.default_rng(42).normal(0, 1, 100)
    n_trials = np.random.default_rng(42).normal(0, 1, 100)
    sampler = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_optuna(objective, n_trials, sampler)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmoptn_edge():
    """Test edge cases."""
    objective = np.random.default_rng(42).normal(0, 1, 100)
    n_trials = np.random.default_rng(42).normal(0, 1, 100)
    sampler = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_optuna(objective, n_trials, sampler)
    assert isinstance(result, dict)
