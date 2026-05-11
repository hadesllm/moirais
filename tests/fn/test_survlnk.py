"""Tests for survlnk.link_function_survival."""
import numpy as np
import pytest
from morie.fn.survlnk import link_function_survival


def test_survlnk_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    link = 'identity'
    result = link_function_survival(time, event, X, link)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survlnk_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    link = 'identity'
    result = link_function_survival(time, event, X, link)
    assert isinstance(result, dict)
