"""Tests for grwpc.geron_wordpiece_tokenizer_score."""
import numpy as np
import pytest
from moirais.fn.grwpc import geron_wordpiece_tokenizer_score


def test_grwpc_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    pairs = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_wordpiece_tokenizer_score(counts, pairs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grwpc_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    pairs = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_wordpiece_tokenizer_score(counts, pairs)
    assert isinstance(result, dict)
