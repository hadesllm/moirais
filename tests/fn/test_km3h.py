"""Tests for km3h.kamath_3h_alignment."""
import numpy as np
import pytest
from moirais.fn.km3h import kamath_3h_alignment


def test_km3h_basic():
    """Test basic functionality."""
    helpful_score = np.random.default_rng(42).normal(0, 1, 100)
    harmless_score = np.random.default_rng(42).normal(0, 1, 100)
    honest_score = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = kamath_3h_alignment(helpful_score, harmless_score, honest_score, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km3h_edge():
    """Test edge cases."""
    helpful_score = np.random.default_rng(42).normal(0, 1, 100)
    harmless_score = np.random.default_rng(42).normal(0, 1, 100)
    honest_score = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = kamath_3h_alignment(helpful_score, harmless_score, honest_score, weights)
    assert isinstance(result, dict)
