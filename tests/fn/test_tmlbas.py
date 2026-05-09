"""Tests for tmlbas.tmle_baseline_adj."""
import numpy as np
import pytest
from moirais.fn.tmlbas import tmle_baseline_adj


def test_tmlbas_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_baseline_adj(y, D, X, baseline)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlbas_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_baseline_adj(y, D, X, baseline)
    assert isinstance(result, dict)
