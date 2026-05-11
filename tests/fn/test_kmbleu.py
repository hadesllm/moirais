"""Tests for kmbleu.kamath_bleu_score."""
import numpy as np
import pytest
from morie.fn.kmbleu import kamath_bleu_score


def test_kmbleu_basic():
    """Test basic functionality."""
    hypothesis = np.random.default_rng(42).normal(0, 1, 100)
    references = np.random.default_rng(42).normal(0, 1, 100)
    max_n = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_bleu_score(hypothesis, references, max_n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmbleu_edge():
    """Test edge cases."""
    hypothesis = np.random.default_rng(42).normal(0, 1, 100)
    references = np.random.default_rng(42).normal(0, 1, 100)
    max_n = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_bleu_score(hypothesis, references, max_n)
    assert isinstance(result, dict)
