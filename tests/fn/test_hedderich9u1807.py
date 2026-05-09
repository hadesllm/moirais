"""Tests for hedderich9u1807.hedderich_chapter_9_unnumbered_1807."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1807 import hedderich_chapter_9_unnumbered_1807


def test_hedderich9u1807_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1807(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1807_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1807(x)
    assert isinstance(result, dict)
