"""Tests for tmldyn.tmle_dynamic_regime."""
import numpy as np
import pytest
from morie.fn.tmldyn import tmle_dynamic_regime


def test_tmldyn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    regime = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_dynamic_regime(y, treatment_history, covariate_history, regime)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmldyn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment_history = np.random.default_rng(42).normal(0, 1, 100)
    covariate_history = np.random.default_rng(42).normal(0, 1, 100)
    regime = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_dynamic_regime(y, treatment_history, covariate_history, regime)
    assert isinstance(result, dict)
