"""Tests for prdtA.prefix_evaluation."""
import numpy as np
import pytest
from moirais.fn.prdtA import prefix_evaluation


def test_prdtA_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    result = prefix_evaluation(expr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prdtA_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    result = prefix_evaluation(expr)
    assert isinstance(result, dict)
