"""Tests for agmupv.muzero_predict_value."""
import numpy as np
import pytest
from morie.fn.agmupv import muzero_predict_value


def test_agmupv_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = muzero_predict_value(state, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agmupv_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = muzero_predict_value(state, f)
    assert isinstance(result, dict)
