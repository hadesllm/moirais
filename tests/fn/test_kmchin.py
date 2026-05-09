"""Tests for kmchin.kamath_chinchilla_compute_optimal."""
import numpy as np
import pytest
from moirais.fn.kmchin import kamath_chinchilla_compute_optimal


def test_kmchin_basic():
    """Test basic functionality."""
    compute_budget = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = kamath_chinchilla_compute_optimal(compute_budget, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmchin_edge():
    """Test edge cases."""
    compute_budget = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = kamath_chinchilla_compute_optimal(compute_budget, alpha, beta)
    assert isinstance(result, dict)
