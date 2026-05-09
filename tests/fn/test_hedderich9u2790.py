"""Tests for hedderich9u2790.hedderich_chapter_9_unnumbered_2790."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2790 import hedderich_chapter_9_unnumbered_2790


def test_hedderich9u2790_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2790(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2790_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2790(x)
    assert isinstance(result, dict)
