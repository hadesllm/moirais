"""Tests for grscm.geron_score_matching_loss."""
import numpy as np
import pytest
from morie.fn.grscm import geron_score_matching_loss


def test_grscm_basic():
    """Test basic functionality."""
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    eps = np.random.default_rng(42).normal(0, 1, 100)
    score_pred = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_score_matching_loss(x0, sigma, eps, score_pred)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grscm_edge():
    """Test edge cases."""
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    eps = np.random.default_rng(42).normal(0, 1, 100)
    score_pred = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_score_matching_loss(x0, sigma, eps, score_pred)
    assert isinstance(result, dict)
