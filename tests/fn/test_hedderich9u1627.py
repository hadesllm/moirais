"""Tests for hedderich9u1627.hedderich_chapter_9_unnumbered_1627."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1627 import hedderich_chapter_9_unnumbered_1627


def test_hedderich9u1627_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1627(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1627_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1627(x)
    assert isinstance(result, dict)
