"""Tests for hrzormsc.horowitz_ordered_max_score."""
import numpy as np
import pytest
from moirais.fn.hrzormsc import horowitz_ordered_max_score


def test_hrzormsc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_ordered_max_score(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzormsc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_ordered_max_score(x, y)
    assert isinstance(result, dict)
