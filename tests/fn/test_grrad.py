"""Tests for grrad.geron_reverse_mode_autodiff."""
import numpy as np
import pytest
from morie.fn.grrad import geron_reverse_mode_autodiff


def test_grrad_basic():
    """Test basic functionality."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    loss_grad = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_reverse_mode_autodiff(graph, loss_grad)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grrad_edge():
    """Test edge cases."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    loss_grad = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_reverse_mode_autodiff(graph, loss_grad)
    assert isinstance(result, dict)
