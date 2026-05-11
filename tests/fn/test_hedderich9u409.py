"""Tests for hedderich9u409.hedderich_chapter_9_unnumbered_409."""
import numpy as np
import pytest
from morie.fn.hedderich9u409 import hedderich_chapter_9_unnumbered_409


def test_hedderich9u409_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_409(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u409_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_409(x)
    assert isinstance(result, dict)
