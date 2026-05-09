"""Tests for kmgrnd.kamath_groundedness_reward."""
import numpy as np
import pytest
from moirais.fn.kmgrnd import kamath_groundedness_reward


def test_kmgrnd_basic():
    """Test basic functionality."""
    y_tokens = np.random.default_rng(42).normal(0, 1, 100)
    ctx_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_groundedness_reward(y_tokens, ctx_tokens)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmgrnd_edge():
    """Test edge cases."""
    y_tokens = np.random.default_rng(42).normal(0, 1, 100)
    ctx_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_groundedness_reward(y_tokens, ctx_tokens)
    assert isinstance(result, dict)
