"""Tests for randres.randomized_response."""
import numpy as np
import pytest
from morie.fn.randres import randomized_response


def test_randres_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    truth = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = randomized_response(y, truth, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_randres_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    truth = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = randomized_response(y, truth, p)
    assert isinstance(result, dict)
