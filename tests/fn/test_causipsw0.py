"""Tests for causipsw0.causal_iptw_atoweights."""
import numpy as np
import pytest
from morie.fn.causipsw0 import causal_iptw_atoweights


def test_causipsw0_basic():
    """Test basic functionality."""
    treat = np.random.default_rng(42).normal(0, 1, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_iptw_atoweights(treat, ps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causipsw0_edge():
    """Test edge cases."""
    treat = np.random.default_rng(42).normal(0, 1, 100)
    ps = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_iptw_atoweights(treat, ps)
    assert isinstance(result, dict)
