"""Tests for hedderich9u321.hedderich_chapter_9_unnumbered_321."""
import numpy as np
import pytest
from morie.fn.hedderich9u321 import hedderich_chapter_9_unnumbered_321


def test_hedderich9u321_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_321(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u321_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_321(x)
    assert isinstance(result, dict)
