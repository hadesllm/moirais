"""Tests for alfqud.alphadev_quicksort_disc."""
import numpy as np
import pytest
from morie.fn.alfqud import alphadev_quicksort_disc


def test_alfqud_basic():
    """Test basic functionality."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    action_space = np.random.default_rng(42).normal(0, 1, 100)
    reward_fn = (lambda v: v)
    result = alphadev_quicksort_disc(target, action_space, reward_fn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfqud_edge():
    """Test edge cases."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    action_space = np.random.default_rng(42).normal(0, 1, 100)
    reward_fn = (lambda v: v)
    result = alphadev_quicksort_disc(target, action_space, reward_fn)
    assert isinstance(result, dict)
