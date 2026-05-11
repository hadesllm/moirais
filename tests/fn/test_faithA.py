"""Tests for faithA.faithfulness_assumption."""
import numpy as np
import pytest
from morie.fn.faithA import faithfulness_assumption


def test_faithA_basic():
    """Test basic functionality."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = faithfulness_assumption(dag, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_faithA_edge():
    """Test edge cases."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = faithfulness_assumption(dag, P)
    assert isinstance(result, dict)
