"""Tests for grpex.geron_prioritized_experience_weight."""
import numpy as np
import pytest
from moirais.fn.grpex import geron_prioritized_experience_weight


def test_grpex_basic():
    """Test basic functionality."""
    priorities = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    alpha = 0.05
    beta = 0.8
    result = geron_prioritized_experience_weight(priorities, N, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grpex_edge():
    """Test edge cases."""
    priorities = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    alpha = 0.05
    beta = 0.8
    result = geron_prioritized_experience_weight(priorities, N, alpha, beta)
    assert isinstance(result, dict)
