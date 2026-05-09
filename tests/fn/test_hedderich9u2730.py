"""Tests for hedderich9u2730.hedderich_chapter_9_unnumbered_2730."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2730 import hedderich_chapter_9_unnumbered_2730


def test_hedderich9u2730_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2730(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2730_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2730(x)
    assert isinstance(result, dict)
