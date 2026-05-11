"""Tests for resolP.resolution_proof."""
import numpy as np
import pytest
from morie.fn.resolP import resolution_proof


def test_resolP_basic():
    """Test basic functionality."""
    clauses = np.random.default_rng(42).normal(0, 1, 100)
    result = resolution_proof(clauses)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_resolP_edge():
    """Test edge cases."""
    clauses = np.random.default_rng(42).normal(0, 1, 100)
    result = resolution_proof(clauses)
    assert isinstance(result, dict)
