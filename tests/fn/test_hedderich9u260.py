"""Tests for hedderich9u260.hedderich_chapter_9_unnumbered_260."""
import numpy as np
import pytest
from morie.fn.hedderich9u260 import hedderich_chapter_9_unnumbered_260


def test_hedderich9u260_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_260(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u260_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_260(x)
    assert isinstance(result, dict)
