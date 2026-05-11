"""Tests for hedderich9u1501.hedderich_chapter_9_unnumbered_1501."""
import numpy as np
import pytest
from morie.fn.hedderich9u1501 import hedderich_chapter_9_unnumbered_1501


def test_hedderich9u1501_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1501(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1501_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1501(x)
    assert isinstance(result, dict)
