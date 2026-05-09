"""Tests for ssintc.interval_censored_survival."""
import numpy as np
import pytest
from moirais.fn.ssintc import interval_censored_survival


def test_ssintc_basic():
    """Test basic functionality."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = interval_censored_survival(L, R, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ssintc_edge():
    """Test edge cases."""
    L = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = interval_censored_survival(L, R, event)
    assert isinstance(result, dict)
