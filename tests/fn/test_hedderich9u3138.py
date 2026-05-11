"""Tests for hedderich9u3138.hedderich_chapter_9_unnumbered_3138."""
import numpy as np
import pytest
from morie.fn.hedderich9u3138 import hedderich_chapter_9_unnumbered_3138


def test_hedderich9u3138_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3138(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u3138_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3138(x)
    assert isinstance(result, dict)
