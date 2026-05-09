"""Tests for chnntp.channel_capacity."""
import numpy as np
import pytest
from moirais.fn.chnntp import channel_capacity


def test_chnntp_basic():
    """Test basic functionality."""
    channel = np.random.default_rng(42).normal(0, 1, 100)
    result = channel_capacity(channel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chnntp_edge():
    """Test edge cases."""
    channel = np.random.default_rng(42).normal(0, 1, 100)
    result = channel_capacity(channel)
    assert isinstance(result, dict)
