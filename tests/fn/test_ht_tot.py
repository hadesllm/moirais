"""Tests for moirais.fn.ht_tot — Horvitz-Thompson total estimator."""

import math

import numpy as np
import pytest

from moirais.fn.ht_tot import horvitz_thompson_total


def test_returns_dict_with_keys():
    """horvitz_thompson_total returns a dict with total, se, ci_lower, ci_upper."""
    rng = np.random.default_rng(42)
    y = rng.uniform(1, 10, size=50)
    pi = rng.uniform(0.1, 0.9, size=50)
    result = horvitz_thompson_total(y, pi)
    assert isinstance(result, dict)
    for key in ("total", "se", "ci_lower", "ci_upper"):
        assert key in result


def test_total_positive_for_positive_data():
    """HT total should be positive when all y_i > 0."""
    rng = np.random.default_rng(42)
    y = rng.uniform(1, 10, size=100)
    pi = rng.uniform(0.1, 0.5, size=100)
    result = horvitz_thompson_total(y, pi)
    assert result["total"] > 0


def test_se_is_finite():
    """SE should be finite and non-negative."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(50)
    pi = rng.uniform(0.2, 0.8, size=50)
    result = horvitz_thompson_total(y, pi)
    assert math.isfinite(result["se"])
    assert result["se"] >= 0


def test_invalid_inclusion_probs_raises():
    """Inclusion probabilities outside (0, 1] should raise ValueError."""
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="inclusion_probs"):
        horvitz_thompson_total(y, np.array([0.0, 0.5, 0.5]))
    with pytest.raises(ValueError, match="inclusion_probs"):
        horvitz_thompson_total(y, np.array([0.5, 1.5, 0.5]))


def test_length_mismatch_raises():
    """y and inclusion_probs of different lengths should raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        horvitz_thompson_total(np.array([1, 2]), np.array([0.5]))
