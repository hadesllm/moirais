"""Tests for hedderich9u1201.hedderich_chapter_9_unnumbered_1201."""
import numpy as np
import pytest
from morie.fn.hedderich9u1201 import hedderich_chapter_9_unnumbered_1201


def test_hedderich9u1201_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1201(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1201_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1201(x)
    assert isinstance(result, dict)
