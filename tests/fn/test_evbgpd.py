"""Tests for evbgpd.evt_bayes_gpd."""
import numpy as np
import pytest
from morie.fn.evbgpd import evt_bayes_gpd


def test_evbgpd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_iter = 50
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_bayes_gpd(y, n_iter, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evbgpd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_iter = 50
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_bayes_gpd(y, n_iter, prior)
    assert isinstance(result, dict)
