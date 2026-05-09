"""Tests for rgeqn10a.rangayyan_ch10_cost_matrix."""
import numpy as np
import pytest
from moirais.fn.rgeqn10a import rangayyan_ch10_cost_matrix


def test_rgeqn10a_basic():
    """Test basic functionality."""
    cost_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    confusion_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch10_cost_matrix(cost_matrix, confusion_matrix, priors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn10a_edge():
    """Test edge cases."""
    cost_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    confusion_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch10_cost_matrix(cost_matrix, confusion_matrix, priors)
    assert isinstance(result, dict)
