"""Tests for rlhfd.rlhf_reward."""
import numpy as np
import pytest
from moirais.fn.rlhfd import rlhf_reward


def test_rlhfd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rlhf_reward(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rlhfd_edge():
    """Test edge cases."""
    result = rlhf_reward(np.array([42.0]))
    assert result['n'] == 1
