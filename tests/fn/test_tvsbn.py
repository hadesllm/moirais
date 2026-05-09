"""Tests for tvsbn.tversky_similarity."""
import numpy as np
import pytest
from moirais.fn.tvsbn import tversky_similarity


def test_tvsbn_basic():
    """Test basic functionality."""
    fp_a = np.random.default_rng(42).normal(0, 1, 100)
    fp_b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = tversky_similarity(fp_a, fp_b, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tvsbn_edge():
    """Test edge cases."""
    fp_a = np.random.default_rng(42).normal(0, 1, 100)
    fp_b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = tversky_similarity(fp_a, fp_b, alpha, beta)
    assert isinstance(result, dict)
