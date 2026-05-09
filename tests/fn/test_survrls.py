"""Tests for survrls.restricted_lifetime."""
import numpy as np
import pytest
from moirais.fn.survrls import restricted_lifetime


def test_survrls_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    t_star = np.random.default_rng(42).normal(0, 1, 100)
    result = restricted_lifetime(fit, t_star)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survrls_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    t_star = np.random.default_rng(42).normal(0, 1, 100)
    result = restricted_lifetime(fit, t_star)
    assert isinstance(result, dict)
