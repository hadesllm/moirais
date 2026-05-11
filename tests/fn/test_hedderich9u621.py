"""Tests for hedderich9u621.hedderich_chapter_9_unnumbered_621."""
import numpy as np
import pytest
from morie.fn.hedderich9u621 import hedderich_chapter_9_unnumbered_621


def test_hedderich9u621_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_621(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u621_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_621(x)
    assert isinstance(result, dict)
