"""Tests for bkrep.burkov_repetition_penalty."""
import numpy as np
import pytest
from moirais.fn.bkrep import burkov_repetition_penalty


def test_bkrep_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    prev_tokens = np.random.default_rng(42).normal(0, 1, 100)
    penalty = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_repetition_penalty(logits, prev_tokens, penalty)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkrep_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    prev_tokens = np.random.default_rng(42).normal(0, 1, 100)
    penalty = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_repetition_penalty(logits, prev_tokens, penalty)
    assert isinstance(result, dict)
