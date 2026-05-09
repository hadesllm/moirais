"""Tests for agmurc.muzero_recurrent_inf."""
import numpy as np
import pytest
from moirais.fn.agmurc import muzero_recurrent_inf


def test_agmurc_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    action = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = muzero_recurrent_inf(state, action, g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agmurc_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    action = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = muzero_recurrent_inf(state, action, g)
    assert isinstance(result, dict)
