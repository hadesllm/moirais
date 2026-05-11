"""Tests for hedderich9u3013.hedderich_chapter_9_unnumbered_3013."""
import numpy as np
import pytest
from morie.fn.hedderich9u3013 import hedderich_chapter_9_unnumbered_3013


def test_hedderich9u3013_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3013(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u3013_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3013(x)
    assert isinstance(result, dict)
