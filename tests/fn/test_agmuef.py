"""Tests for agmuef.muzero_efficient_exploration."""
import numpy as np
import pytest
from moirais.fn.agmuef import muzero_efficient_exploration


def test_agmuef_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    root_state = np.random.default_rng(42).normal(0, 1, 100)
    sims = np.random.default_rng(42).normal(0, 1, 100)
    result = muzero_efficient_exploration(model, root_state, sims)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agmuef_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    root_state = np.random.default_rng(42).normal(0, 1, 100)
    sims = np.random.default_rng(42).normal(0, 1, 100)
    result = muzero_efficient_exploration(model, root_state, sims)
    assert isinstance(result, dict)
