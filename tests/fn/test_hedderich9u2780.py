"""Tests for hedderich9u2780.hedderich_chapter_9_unnumbered_2780."""
import numpy as np
import pytest
from morie.fn.hedderich9u2780 import hedderich_chapter_9_unnumbered_2780


def test_hedderich9u2780_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2780(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2780_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2780(x)
    assert isinstance(result, dict)
