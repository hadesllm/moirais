"""Tests for rouge.rouge."""
import numpy as np
import pytest
from moirais.fn.rouge import rouge


def test_rouge_basic():
    """Test basic functionality."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    kind = np.random.default_rng(42).normal(0, 1, 100)
    result = rouge(candidate, reference, kind)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rouge_edge():
    """Test edge cases."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    kind = np.random.default_rng(42).normal(0, 1, 100)
    result = rouge(candidate, reference, kind)
    assert isinstance(result, dict)
