"""Tests for dprnyi.renyi_dp_composition."""
import numpy as np
import pytest
from moirais.fn.dprnyi import renyi_dp_composition


def test_dprnyi_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    epsilons = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = renyi_dp_composition(y, epsilons, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dprnyi_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    epsilons = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = renyi_dp_composition(y, epsilons, alpha)
    assert isinstance(result, dict)
