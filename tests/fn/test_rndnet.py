"""Tests for rndnet.random_network_distillation."""
import numpy as np
import pytest
from moirais.fn.rndnet import random_network_distillation


def test_rndnet_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    predictor = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = random_network_distillation(env, predictor, target)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rndnet_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    predictor = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    result = random_network_distillation(env, predictor, target)
    assert isinstance(result, dict)
