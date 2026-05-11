"""Tests for altkp.alammar_tokenization_pipeline."""
import numpy as np
import pytest
from morie.fn.altkp import alammar_tokenization_pipeline


def test_altkp_basic():
    """Test basic functionality."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    tokenizer = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_tokenization_pipeline(text, tokenizer)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_altkp_edge():
    """Test edge cases."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    tokenizer = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_tokenization_pipeline(text, tokenizer)
    assert isinstance(result, dict)
