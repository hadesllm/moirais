"""Tests for npbsr.np_bayes_survival."""
import numpy as np
import pytest
from morie.fn.npbsr import np_bayes_survival


def test_npbsr_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = np_bayes_survival(time, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_npbsr_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = np_bayes_survival(time, event)
    assert isinstance(result, dict)
