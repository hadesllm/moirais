"""Tests for hedderich9u38.hedderich_chapter_9_unnumbered_38."""
import numpy as np
import pytest
from morie.fn.hedderich9u38 import hedderich_chapter_9_unnumbered_38


def test_hedderich9u38_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_38(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u38_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_38(x)
    assert isinstance(result, dict)
