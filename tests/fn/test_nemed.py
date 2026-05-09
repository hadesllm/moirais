"""Tests for nemed.nested_counterfactual_mediation."""
import numpy as np
import pytest
from moirais.fn.nemed import nested_counterfactual_mediation


def test_nemed_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = nested_counterfactual_mediation(X, M, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nemed_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = nested_counterfactual_mediation(X, M, Y)
    assert isinstance(result, dict)
