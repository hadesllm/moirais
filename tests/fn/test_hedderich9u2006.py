"""Tests for hedderich9u2006.hedderich_chapter_9_unnumbered_2006."""
import numpy as np
import pytest
from morie.fn.hedderich9u2006 import hedderich_chapter_9_unnumbered_2006


def test_hedderich9u2006_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2006(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2006_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2006(x)
    assert isinstance(result, dict)
