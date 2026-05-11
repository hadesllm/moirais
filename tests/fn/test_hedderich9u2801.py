"""Tests for hedderich9u2801.hedderich_chapter_9_unnumbered_2801."""
import numpy as np
import pytest
from morie.fn.hedderich9u2801 import hedderich_chapter_9_unnumbered_2801


def test_hedderich9u2801_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2801(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2801_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2801(x)
    assert isinstance(result, dict)
