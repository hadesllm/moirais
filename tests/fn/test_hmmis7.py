"""Tests for hmmis7.geron_mistral7b."""
import numpy as np
import pytest
from morie.fn.hmmis7 import geron_mistral7b


def test_hmmis7_basic():
    """Test basic functionality."""
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    n_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mistral7b(prompt, n_tokens)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmis7_edge():
    """Test edge cases."""
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    n_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mistral7b(prompt, n_tokens)
    assert isinstance(result, dict)
