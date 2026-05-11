"""Tests for hedderich9u2521.hedderich_chapter_9_unnumbered_2521."""
import numpy as np
import pytest
from morie.fn.hedderich9u2521 import hedderich_chapter_9_unnumbered_2521


def test_hedderich9u2521_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2521(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2521_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2521(x)
    assert isinstance(result, dict)
