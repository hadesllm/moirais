"""Tests for aitdrf.dirichlet_fit_mom."""
import numpy as np
import pytest
from moirais.fn.aitdrf import dirichlet_fit_mom


def test_aitdrf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dirichlet_fit_mom(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitdrf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dirichlet_fit_mom(X)
    assert isinstance(result, dict)
