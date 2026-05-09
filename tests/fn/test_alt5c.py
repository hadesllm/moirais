"""Tests for alt5c.alammar_t5_text_to_text_classify."""
import numpy as np
import pytest
from moirais.fn.alt5c import alammar_t5_text_to_text_classify


def test_alt5c_basic():
    """Test basic functionality."""
    input = np.random.default_rng(42).normal(0, 1, 100)
    label_tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_t5_text_to_text_classify(input, label_tokens, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alt5c_edge():
    """Test edge cases."""
    input = np.random.default_rng(42).normal(0, 1, 100)
    label_tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_t5_text_to_text_classify(input, label_tokens, model)
    assert isinstance(result, dict)
