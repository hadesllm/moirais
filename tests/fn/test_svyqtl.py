"""Tests for svyqtl.survey_quantile."""
import numpy as np
import pytest
from morie.fn.svyqtl import survey_quantile


def test_svyqtl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = survey_quantile(y, weights, quantile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svyqtl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = survey_quantile(y, weights, quantile)
    assert isinstance(result, dict)
