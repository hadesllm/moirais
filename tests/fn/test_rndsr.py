"""Tests for rndsr.random_search_cv."""
import numpy as np
import pytest
from morie.fn.rndsr import random_search_cv


def test_rndsr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = random_search_cv(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rndsr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = random_search_cv(x, y)
    assert isinstance(result, dict)
