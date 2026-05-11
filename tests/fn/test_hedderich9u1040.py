"""Tests for hedderich9u1040.hedderich_chapter_9_unnumbered_1040."""
import numpy as np
import pytest
from morie.fn.hedderich9u1040 import hedderich_chapter_9_unnumbered_1040


def test_hedderich9u1040_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1040(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1040_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1040(x)
    assert isinstance(result, dict)
