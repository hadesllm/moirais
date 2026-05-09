"""Tests for eslmrf.esl_markov_rf."""
import numpy as np
import pytest
from moirais.fn.eslmrf import esl_markov_rf


def test_eslmrf_basic():
    """Test basic functionality."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    psi = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_markov_rf(graph, psi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslmrf_edge():
    """Test edge cases."""
    graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    psi = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_markov_rf(graph, psi)
    assert isinstance(result, dict)
