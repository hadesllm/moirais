"""Tests for morie.fn.effic — Semiparametric efficiency bound."""

import numpy as np
import pytest

from morie.fn.effic import effic, EfficiencyBoundResult


@pytest.fixture()
def normal_scores():
    rng = np.random.default_rng(42)
    return rng.standard_normal(300)


def test_returns_result_type(normal_scores):
    result = effic(normal_scores)
    assert isinstance(result, EfficiencyBoundResult)


def test_fisher_info_positive(normal_scores):
    result = effic(normal_scores)
    assert result.fisher_info > 0


def test_efficiency_bound_positive(normal_scores):
    result = effic(normal_scores)
    assert result.efficiency_bound > 0


def test_efficient_without_influence(normal_scores):
    result = effic(normal_scores)
    assert result.is_efficient is True
    assert result.efficiency_ratio == 1.0


def test_efficient_with_matching_influence(normal_scores):
    result = effic(normal_scores, influence_values=normal_scores)
    assert result.efficiency_ratio > 0


def test_inefficient_estimator():
    rng = np.random.default_rng(42)
    scores = rng.standard_normal(500)
    noisy = scores + rng.standard_normal(500) * 2.0
    result = effic(scores, influence_values=noisy)
    assert result.efficiency_ratio < 1.0
    assert result.is_efficient is False


def test_n_stored(normal_scores):
    result = effic(normal_scores)
    assert result.n == 300


def test_known_case():
    scores = np.ones(100)
    result = effic(scores)
    assert np.isclose(result.fisher_info, 1.0)
    assert np.isclose(result.efficiency_bound, 1.0)


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        effic(np.array([]))


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        effic(np.array([1.0, 2.0]), influence_values=np.array([1.0]))
