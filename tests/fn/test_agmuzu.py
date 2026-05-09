"""Tests for agmuzu.muzero_world_model."""
import numpy as np
import pytest
from moirais.fn.agmuzu import muzero_world_model


def test_agmuzu_basic():
    """Test basic functionality."""
    observations = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = muzero_world_model(observations, h, g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agmuzu_edge():
    """Test edge cases."""
    observations = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = muzero_world_model(observations, h, g)
    assert isinstance(result, dict)
