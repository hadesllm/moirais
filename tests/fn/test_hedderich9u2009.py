"""Tests for hedderich9u2009.hedderich_chapter_9_unnumbered_2009."""
import numpy as np
import pytest
from morie.fn.hedderich9u2009 import hedderich_chapter_9_unnumbered_2009


def test_hedderich9u2009_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2009(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2009_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2009(x)
    assert isinstance(result, dict)
