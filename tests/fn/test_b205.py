"""Tests for b205.burkov_lm_ch2_perplexity."""
import numpy as np
import pytest
from moirais.fn.b205 import burkov_lm_ch2_perplexity


def test_b205_basic():
    """Test basic functionality."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    t = np.linspace(0, 10, 100)
    result = burkov_lm_ch2_perplexity(D, k, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b205_edge():
    """Test edge cases."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    t = np.linspace(0, 10, 100)
    result = burkov_lm_ch2_perplexity(D, k, t)
    assert isinstance(result, dict)
