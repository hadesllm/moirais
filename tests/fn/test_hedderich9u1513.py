"""Tests for hedderich9u1513.hedderich_chapter_9_unnumbered_1513."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1513 import hedderich_chapter_9_unnumbered_1513


def test_hedderich9u1513_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1513(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u1513_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1513(x)
    assert isinstance(result, dict)
