"""Tests for kmnuc.kamath_nucleus_sampling."""
import numpy as np
import pytest
from moirais.fn.kmnuc import kamath_nucleus_sampling


def test_kmnuc_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = kamath_nucleus_sampling(logits, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmnuc_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = kamath_nucleus_sampling(logits, p)
    assert isinstance(result, dict)
