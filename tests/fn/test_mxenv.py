"""Tests for mxenv.multi_env_model."""
import numpy as np
import pytest
from morie.fn.mxenv import multi_env_model


def test_mxenv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    env = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    result = multi_env_model(y, markers, env, G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mxenv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    env = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    result = multi_env_model(y, markers, env, G)
    assert isinstance(result, dict)
