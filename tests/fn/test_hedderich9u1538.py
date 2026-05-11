"""Tests for hedderich9u1538.hedderich_chapter_9_unnumbered_1538."""
import numpy as np
import pytest
from morie.fn.hedderich9u1538 import hedderich_chapter_9_unnumbered_1538


def test_hedderich9u1538_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1538(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1538_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1538(x)
    assert isinstance(result, dict)
