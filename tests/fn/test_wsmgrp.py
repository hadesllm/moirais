"""Tests for wsmgrp.wasserman_graphical_model."""
import numpy as np
import pytest
from morie.fn.wsmgrp import wasserman_graphical_model


def test_wsmgrp_basic():
    """Test basic functionality."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    psi = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_graphical_model(graph, psi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmgrp_edge():
    """Test edge cases."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    psi = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_graphical_model(graph, psi)
    assert isinstance(result, dict)
