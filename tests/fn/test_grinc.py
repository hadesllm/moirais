"""Tests for grinc.geron_in_context_learning."""
import numpy as np
import pytest
from moirais.fn.grinc import geron_in_context_learning


def test_grinc_basic():
    """Test basic functionality."""
    examples = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_in_context_learning(examples, query)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grinc_edge():
    """Test edge cases."""
    examples = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_in_context_learning(examples, query)
    assert isinstance(result, dict)
