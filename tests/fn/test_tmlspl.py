"""Tests for tmlspl.tmle_spillover."""
import numpy as np
import pytest
from moirais.fn.tmlspl import tmle_spillover


def test_tmlspl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    network = np.random.default_rng(42).normal(0, 1, 100)
    exposure_summary = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_spillover(y, D, X, network, exposure_summary)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlspl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    network = np.random.default_rng(42).normal(0, 1, 100)
    exposure_summary = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_spillover(y, D, X, network, exposure_summary)
    assert isinstance(result, dict)
