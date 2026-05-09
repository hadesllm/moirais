"""Tests for mistr.mistral."""
import numpy as np
import pytest
from moirais.fn.mistr import mistral


def test_mistr_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = mistral(tokens, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mistr_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = mistral(tokens, model)
    assert isinstance(result, dict)
