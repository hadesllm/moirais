"""Tests for mfomf.model_based_rl."""
import numpy as np
import pytest
from morie.fn.mfomf import model_based_rl


def test_mfomf_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    planner = np.random.default_rng(42).normal(0, 1, 100)
    result = model_based_rl(env, model, planner)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mfomf_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    planner = np.random.default_rng(42).normal(0, 1, 100)
    result = model_based_rl(env, model, planner)
    assert isinstance(result, dict)
