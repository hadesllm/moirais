"""Tests for otmot.ot_multimarginal_iter."""
import numpy as np
import pytest
from moirais.fn.otmot import ot_multimarginal_iter


def test_otmot_basic():
    """Test basic functionality."""
    margins = np.random.default_rng(42).normal(0, 1, 100)
    C_tensor = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_multimarginal_iter(margins, C_tensor, epsilon, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmot_edge():
    """Test edge cases."""
    margins = np.random.default_rng(42).normal(0, 1, 100)
    C_tensor = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_multimarginal_iter(margins, C_tensor, epsilon, max_iter)
    assert isinstance(result, dict)
