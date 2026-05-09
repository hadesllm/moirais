"""Tests for kmperp.kamath_perplexity."""
import numpy as np
import pytest
from moirais.fn.kmperp import kamath_perplexity


def test_kmperp_basic():
    """Test basic functionality."""
    log_probs = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_perplexity(log_probs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmperp_edge():
    """Test edge cases."""
    log_probs = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_perplexity(log_probs)
    assert isinstance(result, dict)
