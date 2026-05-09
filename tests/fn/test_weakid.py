"""Tests for weakid.weak_identification_mediation."""
import numpy as np
import pytest
from moirais.fn.weakid import weak_identification_mediation


def test_weakid_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    se_a = np.random.default_rng(42).normal(0, 1, 100)
    se_b = np.random.default_rng(42).normal(0, 1, 100)
    result = weak_identification_mediation(a, b, se_a, se_b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_weakid_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    se_a = np.random.default_rng(42).normal(0, 1, 100)
    se_b = np.random.default_rng(42).normal(0, 1, 100)
    result = weak_identification_mediation(a, b, se_a, se_b)
    assert isinstance(result, dict)
