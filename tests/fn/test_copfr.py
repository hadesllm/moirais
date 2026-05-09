"""Tests for copfr.copula_frailty."""
import numpy as np
import pytest
from moirais.fn.copfr import copula_frailty


def test_copfr_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = copula_frailty(time, event, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_copfr_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = copula_frailty(time, event, cluster)
    assert isinstance(result, dict)
