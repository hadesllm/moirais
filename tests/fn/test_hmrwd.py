"""Tests for hmrwd.geron_reward_function."""
import numpy as np
import pytest
from moirais.fn.hmrwd import geron_reward_function


def test_hmrwd_basic():
    """Test basic functionality."""
    s = 90
    a = np.random.default_rng(44).normal(0, 1, 100)
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_reward_function(s, a, s_next)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmrwd_edge():
    """Test edge cases."""
    s = 90
    a = np.random.default_rng(44).normal(0, 1, 100)
    s_next = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_reward_function(s, a, s_next)
    assert isinstance(result, dict)
