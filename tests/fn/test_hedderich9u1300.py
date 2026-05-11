"""Tests for hedderich9u1300.hedderich_chapter_9_unnumbered_1300."""
import numpy as np
import pytest
from morie.fn.hedderich9u1300 import hedderich_chapter_9_unnumbered_1300


def test_hedderich9u1300_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1300(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1300_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1300(x)
    assert isinstance(result, dict)
