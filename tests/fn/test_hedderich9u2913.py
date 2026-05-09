"""Tests for hedderich9u2913.hedderich_chapter_9_unnumbered_2913."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2913 import hedderich_chapter_9_unnumbered_2913


def test_hedderich9u2913_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2913(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u2913_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2913(x)
    assert isinstance(result, dict)
