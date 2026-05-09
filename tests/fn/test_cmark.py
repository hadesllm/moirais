"""Tests for cmark.causal_markov_condition."""
import numpy as np
import pytest
from moirais.fn.cmark import causal_markov_condition


def test_cmark_basic():
    """Test basic functionality."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_markov_condition(dag, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cmark_edge():
    """Test edge cases."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_markov_condition(dag, P)
    assert isinstance(result, dict)
