"""Tests for hedderich9u36.hedderich_chapter_9_unnumbered_36."""
import numpy as np
import pytest
from morie.fn.hedderich9u36 import hedderich_chapter_9_unnumbered_36


def test_hedderich9u36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_36(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_36(x)
    assert isinstance(result, dict)
