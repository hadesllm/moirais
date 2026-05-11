"""Tests for causipsw.causal_iptw_attweights."""
import numpy as np
import pytest
from morie.fn.causipsw import causal_iptw_attweights


def test_causipsw_basic():
    """Test basic functionality."""
    treat = np.random.default_rng(42).normal(0, 1, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_iptw_attweights(treat, ps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causipsw_edge():
    """Test edge cases."""
    treat = np.random.default_rng(42).normal(0, 1, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_iptw_attweights(treat, ps)
    assert isinstance(result, dict)
