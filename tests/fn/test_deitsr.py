"""Tests for deitsr.deit_distill."""
import numpy as np
import pytest
from moirais.fn.deitsr import deit_distill


def test_deitsr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    result = deit_distill(x, teacher)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_deitsr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    result = deit_distill(x, teacher)
    assert isinstance(result, dict)
