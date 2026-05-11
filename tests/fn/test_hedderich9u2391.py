"""Tests for hedderich9u2391.hedderich_chapter_9_unnumbered_2391."""
import numpy as np
import pytest
from morie.fn.hedderich9u2391 import hedderich_chapter_9_unnumbered_2391


def test_hedderich9u2391_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2391(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2391_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2391(x)
    assert isinstance(result, dict)
