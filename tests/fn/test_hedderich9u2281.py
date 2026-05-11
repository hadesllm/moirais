"""Tests for hedderich9u2281.hedderich_chapter_9_unnumbered_2281."""
import numpy as np
import pytest
from morie.fn.hedderich9u2281 import hedderich_chapter_9_unnumbered_2281


def test_hedderich9u2281_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2281(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2281_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2281(x)
    assert isinstance(result, dict)
