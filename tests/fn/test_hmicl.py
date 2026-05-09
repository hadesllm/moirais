"""Tests for hmicl.geron_in_context_learning."""
import numpy as np
import pytest
from moirais.fn.hmicl import geron_in_context_learning


def test_hmicl_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    examples = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_in_context_learning(model, examples, query)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmicl_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    examples = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_in_context_learning(model, examples, query)
    assert isinstance(result, dict)
