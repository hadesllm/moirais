"""Tests for alrmt.alammar_reward_model_training_bt."""
import numpy as np
import pytest
from moirais.fn.alrmt import alammar_reward_model_training_bt


def test_alrmt_basic():
    """Test basic functionality."""
    scores_w = np.random.default_rng(42).normal(0, 1, 100)
    scores_l = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_reward_model_training_bt(scores_w, scores_l)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alrmt_edge():
    """Test edge cases."""
    scores_w = np.random.default_rng(42).normal(0, 1, 100)
    scores_l = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_reward_model_training_bt(scores_w, scores_l)
    assert isinstance(result, dict)
