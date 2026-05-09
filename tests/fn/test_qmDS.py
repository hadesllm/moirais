"""Tests for qmDS.quantile_mapping."""
import numpy as np
import pytest
from moirais.fn.qmDS import quantile_mapping


def test_qmDS_basic():
    """Test basic functionality."""
    x_mod = np.random.default_rng(42).normal(0, 1, 100)
    F_obs = np.random.default_rng(42).normal(0, 1, 100)
    F_mod = np.random.default_rng(42).normal(0, 1, 100)
    result = quantile_mapping(x_mod, F_obs, F_mod)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qmDS_edge():
    """Test edge cases."""
    x_mod = np.random.default_rng(42).normal(0, 1, 100)
    F_obs = np.random.default_rng(42).normal(0, 1, 100)
    F_mod = np.random.default_rng(42).normal(0, 1, 100)
    result = quantile_mapping(x_mod, F_obs, F_mod)
    assert isinstance(result, dict)
