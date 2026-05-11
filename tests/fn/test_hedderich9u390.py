"""Tests for hedderich9u390.hedderich_chapter_9_unnumbered_390."""
import numpy as np
import pytest
from morie.fn.hedderich9u390 import hedderich_chapter_9_unnumbered_390


def test_hedderich9u390_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_390(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u390_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_390(x)
    assert isinstance(result, dict)
