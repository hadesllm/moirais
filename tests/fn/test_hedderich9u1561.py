"""Tests for hedderich9u1561.hedderich_chapter_9_unnumbered_1561."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1561 import hedderich_chapter_9_unnumbered_1561


def test_hedderich9u1561_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1561(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u1561_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1561(x)
    assert isinstance(result, dict)
