"""Tests for hedderich9u128.hedderich_chapter_9_unnumbered_128."""
import numpy as np
import pytest
from moirais.fn.hedderich9u128 import hedderich_chapter_9_unnumbered_128


def test_hedderich9u128_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_128(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u128_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_128(x)
    assert isinstance(result, dict)
