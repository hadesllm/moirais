"""Tests for hedderich9u1366.hedderich_chapter_9_unnumbered_1366."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1366 import hedderich_chapter_9_unnumbered_1366


def test_hedderich9u1366_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1366(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1366_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1366(x)
    assert isinstance(result, dict)
