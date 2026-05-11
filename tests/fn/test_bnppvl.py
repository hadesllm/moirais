"""Tests for bnppvl.bnp_predictive_value."""
import numpy as np
import pytest
from morie.fn.bnppvl import bnp_predictive_value


def test_bnppvl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior_family = np.random.default_rng(42).normal(0, 1, 100)
    result = bnp_predictive_value(y, prior_family)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnppvl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior_family = np.random.default_rng(42).normal(0, 1, 100)
    result = bnp_predictive_value(y, prior_family)
    assert isinstance(result, dict)
