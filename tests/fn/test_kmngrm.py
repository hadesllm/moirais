"""Tests for kmngrm.kamath_ngram_language_model."""
import numpy as np
import pytest
from moirais.fn.kmngrm import kamath_ngram_language_model


def test_kmngrm_basic():
    """Test basic functionality."""
    counts_ngram = np.random.default_rng(42).normal(0, 1, 100)
    counts_prefix = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ngram_language_model(counts_ngram, counts_prefix)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmngrm_edge():
    """Test edge cases."""
    counts_ngram = np.random.default_rng(42).normal(0, 1, 100)
    counts_prefix = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ngram_language_model(counts_ngram, counts_prefix)
    assert isinstance(result, dict)
