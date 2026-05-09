"""Tests for tqoutl.turboquant_outlier_channel_split."""
import numpy as np
import pytest
from moirais.fn.tqoutl import turboquant_outlier_channel_split


def test_tqoutl_basic():
    """Test basic functionality."""
    channels = np.random.default_rng(42).normal(0, 1, 100)
    outlier_threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_outlier_channel_split(channels, outlier_threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqoutl_edge():
    """Test edge cases."""
    channels = np.random.default_rng(42).normal(0, 1, 100)
    outlier_threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_outlier_channel_split(channels, outlier_threshold)
    assert isinstance(result, dict)
