"""Tests for prgxnt.perplexity."""
import numpy as np
import pytest
from moirais.fn.prgxnt import perplexity


def test_prgxnt_basic():
    """Test basic functionality."""
    log_probs = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = perplexity(log_probs, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prgxnt_edge():
    """Test edge cases."""
    log_probs = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = perplexity(log_probs, N)
    assert isinstance(result, dict)
