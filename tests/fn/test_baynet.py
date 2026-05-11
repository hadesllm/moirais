"""Tests for baynet.bayes_network."""
import numpy as np
import pytest
from morie.fn.baynet import bayes_network


def test_baynet_basic():
    """Test basic functionality."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    cpts = np.random.default_rng(42).normal(0, 1, 100)
    evidence = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_network(graph, cpts, evidence, query)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baynet_edge():
    """Test edge cases."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    cpts = np.random.default_rng(42).normal(0, 1, 100)
    evidence = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_network(graph, cpts, evidence, query)
    assert isinstance(result, dict)
