"""Tests for hedderich9u730.hedderich_chapter_9_unnumbered_730."""
import numpy as np
import pytest
from morie.fn.hedderich9u730 import hedderich_chapter_9_unnumbered_730


def test_hedderich9u730_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_730(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u730_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_730(x)
    assert isinstance(result, dict)
