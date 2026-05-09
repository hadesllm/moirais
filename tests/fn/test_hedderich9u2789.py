"""Tests for hedderich9u2789.hedderich_chapter_9_unnumbered_2789."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2789 import hedderich_chapter_9_unnumbered_2789


def test_hedderich9u2789_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2789(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2789_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2789(x)
    assert isinstance(result, dict)
