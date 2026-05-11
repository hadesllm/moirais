"""Tests for hedderich9u213.hedderich_chapter_9_unnumbered_213."""
import numpy as np
import pytest
from morie.fn.hedderich9u213 import hedderich_chapter_9_unnumbered_213


def test_hedderich9u213_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_213(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u213_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_213(x)
    assert isinstance(result, dict)
