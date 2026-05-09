"""Tests for clyfr.clayton_copula_frailty."""
import numpy as np
import pytest
from moirais.fn.clyfr import clayton_copula_frailty


def test_clyfr_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = clayton_copula_frailty(time, event, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clyfr_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = clayton_copula_frailty(time, event, cluster)
    assert isinstance(result, dict)
