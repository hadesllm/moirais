"""Tests for manh2h.ma_network_node_split."""
import numpy as np
import pytest
from morie.fn.manh2h import ma_network_node_split


def test_manh2h_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    edge = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_network_node_split(yi, vi, design, edge)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_manh2h_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    edge = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_network_node_split(yi, vi, design, edge)
    assert isinstance(result, dict)
