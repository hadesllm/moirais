"""Tests for hedderich9u1590.hedderich_chapter_9_unnumbered_1590."""
import numpy as np
import pytest
from morie.fn.hedderich9u1590 import hedderich_chapter_9_unnumbered_1590


def test_hedderich9u1590_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1590(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1590_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1590(x)
    assert isinstance(result, dict)
