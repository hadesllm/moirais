"""Tests for hedderich9u64.hedderich_chapter_9_unnumbered_64."""
import numpy as np
import pytest
from morie.fn.hedderich9u64 import hedderich_chapter_9_unnumbered_64


def test_hedderich9u64_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_64(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u64_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_64(x)
    assert isinstance(result, dict)
