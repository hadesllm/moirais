"""Tests for hedderich9u3017.hedderich_chapter_9_unnumbered_3017."""
import numpy as np
import pytest
from morie.fn.hedderich9u3017 import hedderich_chapter_9_unnumbered_3017


def test_hedderich9u3017_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3017(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u3017_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3017(x)
    assert isinstance(result, dict)
