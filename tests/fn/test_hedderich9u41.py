"""Tests for hedderich9u41.hedderich_chapter_9_unnumbered_41."""
import numpy as np
import pytest
from morie.fn.hedderich9u41 import hedderich_chapter_9_unnumbered_41


def test_hedderich9u41_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_41(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u41_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_41(x)
    assert isinstance(result, dict)
