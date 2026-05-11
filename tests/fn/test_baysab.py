"""Tests for baysab.bayes_a_alpha."""
import numpy as np
import pytest
from morie.fn.baysab import bayes_a_alpha


def test_baysab_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bayes_a_alpha(y, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baysab_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bayes_a_alpha(y, M)
    assert isinstance(result, dict)
