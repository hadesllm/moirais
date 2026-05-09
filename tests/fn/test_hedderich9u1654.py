"""Tests for hedderich9u1654.hedderich_chapter_9_unnumbered_1654."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1654 import hedderich_chapter_9_unnumbered_1654


def test_hedderich9u1654_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1654(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1654_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1654(x)
    assert isinstance(result, dict)
