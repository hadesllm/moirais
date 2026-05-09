"""Tests for kmrmloss.kamath_reward_model_training_loss."""
import numpy as np
import pytest
from moirais.fn.kmrmloss import kamath_reward_model_training_loss


def test_kmrmloss_basic():
    """Test basic functionality."""
    scores_w = np.random.default_rng(42).normal(0, 1, 100)
    scores_l = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_reward_model_training_loss(scores_w, scores_l)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmrmloss_edge():
    """Test edge cases."""
    scores_w = np.random.default_rng(42).normal(0, 1, 100)
    scores_l = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_reward_model_training_loss(scores_w, scores_l)
    assert isinstance(result, dict)
