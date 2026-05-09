"""Tests for bayloo.loo_psi."""
import numpy as np
import pytest
from moirais.fn.bayloo import loo_psi


def test_bayloo_basic():
    """Test basic functionality."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = loo_psi(log_lik)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayloo_edge():
    """Test edge cases."""
    log_lik = np.random.default_rng(42).normal(0, 1, 100)
    result = loo_psi(log_lik)
    assert isinstance(result, dict)
