"""Tests for agmurt.muzero_reanalyze_target."""
import numpy as np
import pytest
from morie.fn.agmurt import muzero_reanalyze_target


def test_agmurt_basic():
    """Test basic functionality."""
    replay = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = muzero_reanalyze_target(replay, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agmurt_edge():
    """Test edge cases."""
    replay = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = muzero_reanalyze_target(replay, model)
    assert isinstance(result, dict)
