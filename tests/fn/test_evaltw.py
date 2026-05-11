"""Tests for evaltw.e_value_unmeasured_confounding."""
import numpy as np
import pytest
from morie.fn.evaltw import e_value_unmeasured_confounding


def test_evaltw_basic():
    """Test basic functionality."""
    estimate = np.random.default_rng(42).normal(0, 1, 100)
    ci_lower = np.random.default_rng(42).normal(0, 1, 100)
    ci_upper = np.random.default_rng(42).normal(0, 1, 100)
    result = e_value_unmeasured_confounding(estimate, ci_lower, ci_upper)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evaltw_edge():
    """Test edge cases."""
    estimate = np.random.default_rng(42).normal(0, 1, 100)
    ci_lower = np.random.default_rng(42).normal(0, 1, 100)
    ci_upper = np.random.default_rng(42).normal(0, 1, 100)
    result = e_value_unmeasured_confounding(estimate, ci_lower, ci_upper)
    assert isinstance(result, dict)
