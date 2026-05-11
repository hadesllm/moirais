"""Tests for hmvbrt.geron_videobert."""
import numpy as np
import pytest
from morie.fn.hmvbrt import geron_videobert


def test_hmvbrt_basic():
    """Test basic functionality."""
    video_tokens = np.random.default_rng(42).normal(0, 1, 100)
    text_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_videobert(video_tokens, text_tokens)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmvbrt_edge():
    """Test edge cases."""
    video_tokens = np.random.default_rng(42).normal(0, 1, 100)
    text_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_videobert(video_tokens, text_tokens)
    assert isinstance(result, dict)
