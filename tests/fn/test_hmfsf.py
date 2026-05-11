"""Tests for hmfsf.geron_few_shot."""
import numpy as np
import pytest
from morie.fn.hmfsf import geron_few_shot


def test_hmfsf_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    examples = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = geron_few_shot(model, examples, query, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmfsf_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    examples = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = geron_few_shot(model, examples, query, k)
    assert isinstance(result, dict)
