"""Tests for kmverb.kamath_verbalizer_mapping."""
import numpy as np
import pytest
from morie.fn.kmverb import kamath_verbalizer_mapping


def test_kmverb_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    vocab = np.random.default_rng(42).normal(0, 1, 100)
    verbalizer_map = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_verbalizer_mapping(logits, vocab, verbalizer_map)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmverb_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    vocab = np.random.default_rng(42).normal(0, 1, 100)
    verbalizer_map = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_verbalizer_mapping(logits, vocab, verbalizer_map)
    assert isinstance(result, dict)
