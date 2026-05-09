"""Tests for hmgpt3.geron_gpt3."""
import numpy as np
import pytest
from moirais.fn.hmgpt3 import geron_gpt3


def test_hmgpt3_basic():
    """Test basic functionality."""
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    n_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gpt3(prompt, n_tokens)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmgpt3_edge():
    """Test edge cases."""
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    n_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gpt3(prompt, n_tokens)
    assert isinstance(result, dict)
