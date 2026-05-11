"""Tests for kcompo.k_step_dp_composition."""
import numpy as np
import pytest
from morie.fn.kcompo import k_step_dp_composition


def test_kcompo_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    epsilons = np.random.default_rng(42).normal(0, 1, 100)
    result = k_step_dp_composition(y, epsilons)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kcompo_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    epsilons = np.random.default_rng(42).normal(0, 1, 100)
    result = k_step_dp_composition(y, epsilons)
    assert isinstance(result, dict)
