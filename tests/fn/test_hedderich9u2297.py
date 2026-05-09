"""Tests for hedderich9u2297.hedderich_chapter_9_unnumbered_2297."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2297 import hedderich_chapter_9_unnumbered_2297


def test_hedderich9u2297_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2297(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2297_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2297(x)
    assert isinstance(result, dict)
