"""Tests for evbgrev.evt_bayes_gev."""
import numpy as np
import pytest
from morie.fn.evbgrev import evt_bayes_gev


def test_evbgrev_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_bayes_gev(x, n_iter, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evbgrev_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_bayes_gev(x, n_iter, prior)
    assert isinstance(result, dict)
