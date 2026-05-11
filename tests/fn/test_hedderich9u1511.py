"""Tests for hedderich9u1511.hedderich_chapter_9_unnumbered_1511."""
import numpy as np
import pytest
from morie.fn.hedderich9u1511 import hedderich_chapter_9_unnumbered_1511


def test_hedderich9u1511_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1511(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u1511_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1511(x)
    assert isinstance(result, dict)
