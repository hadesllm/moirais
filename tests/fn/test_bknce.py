"""Tests for bknce.burkov_noise_contrastive_estimation."""
import numpy as np
import pytest
from morie.fn.bknce import burkov_noise_contrastive_estimation


def test_bknce_basic():
    """Test basic functionality."""
    data_scores = np.random.default_rng(42).normal(0, 1, 100)
    noise_scores = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    q_data = np.random.default_rng(42).normal(0, 1, 100)
    q_noise = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_noise_contrastive_estimation(data_scores, noise_scores, k, q_data, q_noise)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bknce_edge():
    """Test edge cases."""
    data_scores = np.random.default_rng(42).normal(0, 1, 100)
    noise_scores = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    q_data = np.random.default_rng(42).normal(0, 1, 100)
    q_noise = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_noise_contrastive_estimation(data_scores, noise_scores, k, q_data, q_noise)
    assert isinstance(result, dict)
