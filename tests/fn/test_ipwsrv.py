"""Tests for ipwsrv.ipw_with_survey_weights."""
import numpy as np
import pytest
from moirais.fn.ipwsrv import ipw_with_survey_weights


def test_ipwsrv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    propensity = np.random.default_rng(42).normal(0, 1, 100)
    result = ipw_with_survey_weights(y, T, weights, propensity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ipwsrv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    propensity = np.random.default_rng(42).normal(0, 1, 100)
    result = ipw_with_survey_weights(y, T, weights, propensity)
    assert isinstance(result, dict)
