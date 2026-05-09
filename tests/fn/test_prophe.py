"""Tests for prophe.facebook_prophet."""
import numpy as np
import pytest
from moirais.fn.prophe import facebook_prophet


def test_prophe_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    date = np.random.default_rng(42).normal(0, 1, 100)
    seasonality = np.random.default_rng(42).normal(0, 1, 100)
    result = facebook_prophet(y, date, seasonality)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prophe_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    date = np.random.default_rng(42).normal(0, 1, 100)
    seasonality = np.random.default_rng(42).normal(0, 1, 100)
    result = facebook_prophet(y, date, seasonality)
    assert isinstance(result, dict)
