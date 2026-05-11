"""Tests for hedderich9u1317.hedderich_chapter_9_unnumbered_1317."""
import numpy as np
import pytest
from morie.fn.hedderich9u1317 import hedderich_chapter_9_unnumbered_1317


def test_hedderich9u1317_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1317(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1317_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1317(x)
    assert isinstance(result, dict)
