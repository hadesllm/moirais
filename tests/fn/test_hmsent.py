"""Tests for hmsent.geron_sentiment_analysis."""
import numpy as np
import pytest
from morie.fn.hmsent import geron_sentiment_analysis


def test_hmsent_basic():
    """Test basic functionality."""
    texts = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    tokenizer = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sentiment_analysis(texts, model, tokenizer)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsent_edge():
    """Test edge cases."""
    texts = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    tokenizer = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sentiment_analysis(texts, model, tokenizer)
    assert isinstance(result, dict)
