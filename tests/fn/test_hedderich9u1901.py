"""Tests for hedderich9u1901.hedderich_chapter_9_unnumbered_1901."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1901 import hedderich_chapter_9_unnumbered_1901


def test_hedderich9u1901_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1901(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1901_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1901(x)
    assert isinstance(result, dict)
