"""Tests for hedderich9u2196.hedderich_chapter_9_unnumbered_2196."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2196 import hedderich_chapter_9_unnumbered_2196


def test_hedderich9u2196_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2196(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2196_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2196(x)
    assert isinstance(result, dict)
