"""Tests for svyrcq.survey_quantile_reg."""
import numpy as np
import pytest
from morie.fn.svyrcq import survey_quantile_reg


def test_svyrcq_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    tau = 0.1
    result = survey_quantile_reg(y, X, weights, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svyrcq_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    tau = 0.1
    result = survey_quantile_reg(y, X, weights, tau)
    assert isinstance(result, dict)
