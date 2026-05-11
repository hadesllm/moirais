"""Tests for kmroug.kamath_rouge_n."""
import numpy as np
import pytest
from morie.fn.kmroug import kamath_rouge_n


def test_kmroug_basic():
    """Test basic functionality."""
    hypothesis = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kamath_rouge_n(hypothesis, reference, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmroug_edge():
    """Test edge cases."""
    hypothesis = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kamath_rouge_n(hypothesis, reference, n)
    assert isinstance(result, dict)
