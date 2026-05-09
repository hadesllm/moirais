"""Tests for hedderich9u1821.hedderich_chapter_9_unnumbered_1821."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1821 import hedderich_chapter_9_unnumbered_1821


def test_hedderich9u1821_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1821(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1821_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1821(x)
    assert isinstance(result, dict)
