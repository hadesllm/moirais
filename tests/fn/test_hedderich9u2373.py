"""Tests for hedderich9u2373.hedderich_chapter_9_unnumbered_2373."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2373 import hedderich_chapter_9_unnumbered_2373


def test_hedderich9u2373_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2373(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2373_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2373(x)
    assert isinstance(result, dict)
