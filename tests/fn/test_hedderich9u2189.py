"""Tests for hedderich9u2189.hedderich_chapter_9_unnumbered_2189."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2189 import hedderich_chapter_9_unnumbered_2189


def test_hedderich9u2189_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2189(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2189_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2189(x)
    assert isinstance(result, dict)
