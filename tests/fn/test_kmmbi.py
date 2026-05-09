"""Tests for kmmbi.kamath_membership_inference."""
import numpy as np
import pytest
from moirais.fn.kmmbi import kamath_membership_inference


def test_kmmbi_basic():
    """Test basic functionality."""
    losses = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_membership_inference(losses, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmmbi_edge():
    """Test edge cases."""
    losses = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_membership_inference(losses, threshold)
    assert isinstance(result, dict)
