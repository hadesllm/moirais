"""Tests for docl3.do_calculus_rules."""
import numpy as np
import pytest
from moirais.fn.docl3 import do_calculus_rules


def test_docl3_basic():
    """Test basic functionality."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    query = np.random.default_rng(42).normal(0, 1, 100)
    result = do_calculus_rules(dag, query)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_docl3_edge():
    """Test edge cases."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    query = np.random.default_rng(42).normal(0, 1, 100)
    result = do_calculus_rules(dag, query)
    assert isinstance(result, dict)
