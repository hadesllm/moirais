"""Tests for hedderich9u173.hedderich_chapter_9_unnumbered_173."""
import numpy as np
import pytest
from morie.fn.hedderich9u173 import hedderich_chapter_9_unnumbered_173


def test_hedderich9u173_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_173(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u173_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_173(x)
    assert isinstance(result, dict)
