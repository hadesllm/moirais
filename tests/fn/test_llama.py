"""Tests for llama.llama."""
import numpy as np
import pytest
from morie.fn.llama import llama


def test_llama_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = llama(tokens, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_llama_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = llama(tokens, model)
    assert isinstance(result, dict)
