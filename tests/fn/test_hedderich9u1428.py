"""Tests for hedderich9u1428.hedderich_chapter_9_unnumbered_1428."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1428 import hedderich_chapter_9_unnumbered_1428


def test_hedderich9u1428_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1428(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u1428_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1428(x)
    assert isinstance(result, dict)
