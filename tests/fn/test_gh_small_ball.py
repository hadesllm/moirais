"""Tests for gh_small_ball.ghosal_small_ball_prob."""
import numpy as np
import pytest
from moirais.fn.gh_small_ball import ghosal_small_ball_prob


def test_gh_small_ball_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_small_ball_prob(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_small_ball_edge():
    """Test edge cases."""
    result = ghosal_small_ball_prob(np.array([42.0]))
    assert result['n'] == 1
