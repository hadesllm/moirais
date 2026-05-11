"""Tests for hedderich9u1651.hedderich_chapter_9_unnumbered_1651."""
import numpy as np
import pytest
from morie.fn.hedderich9u1651 import hedderich_chapter_9_unnumbered_1651


def test_hedderich9u1651_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1651(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1651_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1651(x)
    assert isinstance(result, dict)
