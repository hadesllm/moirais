"""Tests for ident.identifiability_conditions."""
import numpy as np
import pytest
from morie.fn.ident import identifiability_conditions


def test_ident_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    result = identifiability_conditions(data, dag)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ident_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    result = identifiability_conditions(data, dag)
    assert isinstance(result, dict)
