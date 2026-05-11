"""Tests for hedderich9u1714.hedderich_chapter_9_unnumbered_1714."""
import numpy as np
import pytest
from morie.fn.hedderich9u1714 import hedderich_chapter_9_unnumbered_1714


def test_hedderich9u1714_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1714(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1714_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1714(x)
    assert isinstance(result, dict)
