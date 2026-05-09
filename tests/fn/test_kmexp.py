"""Tests for kmexp.kamath_memorization_exposure."""
import numpy as np
import pytest
from moirais.fn.kmexp import kamath_memorization_exposure


def test_kmexp_basic():
    """Test basic functionality."""
    canary_ll = np.random.default_rng(42).normal(0, 1, 100)
    candidate_lls = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_memorization_exposure(canary_ll, candidate_lls)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmexp_edge():
    """Test edge cases."""
    canary_ll = np.random.default_rng(42).normal(0, 1, 100)
    candidate_lls = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_memorization_exposure(canary_ll, candidate_lls)
    assert isinstance(result, dict)
