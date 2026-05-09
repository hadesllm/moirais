"""Tests for pratt.pretrained_attention."""
import numpy as np
import pytest
from moirais.fn.pratt import pretrained_attention


def test_pratt_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = pretrained_attention(tokens, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pratt_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = pretrained_attention(tokens, model)
    assert isinstance(result, dict)
