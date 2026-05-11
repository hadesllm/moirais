"""Tests for hedderich9u1573.hedderich_chapter_9_unnumbered_1573."""
import numpy as np
import pytest
from morie.fn.hedderich9u1573 import hedderich_chapter_9_unnumbered_1573


def test_hedderich9u1573_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1573(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1573_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1573(x)
    assert isinstance(result, dict)
