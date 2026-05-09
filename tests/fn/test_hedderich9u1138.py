"""Tests for hedderich9u1138.hedderich_chapter_9_unnumbered_1138."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1138 import hedderich_chapter_9_unnumbered_1138


def test_hedderich9u1138_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1138(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1138_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1138(x)
    assert isinstance(result, dict)
