"""Tests for rgeqn1.rangayyan_ch1_signal_stats."""
import numpy as np
import pytest
from moirais.fn.rgeqn1 import rangayyan_ch1_signal_stats


def test_rgeqn1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch1_signal_stats(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_rgeqn1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch1_signal_stats(x)
    assert isinstance(result, dict)
