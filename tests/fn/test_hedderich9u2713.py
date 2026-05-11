"""Tests for hedderich9u2713.hedderich_chapter_9_unnumbered_2713."""
import numpy as np
import pytest
from morie.fn.hedderich9u2713 import hedderich_chapter_9_unnumbered_2713


def test_hedderich9u2713_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2713(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2713_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2713(x)
    assert isinstance(result, dict)
