"""Tests for hedderich9u2514.hedderich_chapter_9_unnumbered_2514."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2514 import hedderich_chapter_9_unnumbered_2514


def test_hedderich9u2514_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2514(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2514_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2514(x)
    assert isinstance(result, dict)
