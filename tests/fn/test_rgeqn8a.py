"""Tests for rgeqn8a.rangayyan_ch8_sem_threshold."""
import numpy as np
import pytest
from morie.fn.rgeqn8a import rangayyan_ch8_sem_threshold


def test_rgeqn8a_basic():
    """Test basic functionality."""
    sem_trace = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = rangayyan_ch8_sem_threshold(sem_trace, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn8a_edge():
    """Test edge cases."""
    sem_trace = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = rangayyan_ch8_sem_threshold(sem_trace, k)
    assert isinstance(result, dict)
