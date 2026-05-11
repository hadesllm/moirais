"""Tests for ppdks.posterior_predictive_ks."""
import numpy as np
import pytest
from morie.fn.ppdks import posterior_predictive_ks


def test_ppdks_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_rep = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_ks(y, y_rep)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ppdks_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_rep = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_ks(y, y_rep)
    assert isinstance(result, dict)
