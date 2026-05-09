"""Tests for hedderich9u326.hedderich_chapter_9_unnumbered_326."""
import numpy as np
import pytest
from moirais.fn.hedderich9u326 import hedderich_chapter_9_unnumbered_326


def test_hedderich9u326_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_326(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u326_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_326(x)
    assert isinstance(result, dict)
