"""Tests for hedderich9u3153.hedderich_chapter_9_unnumbered_3153."""
import numpy as np
import pytest
from morie.fn.hedderich9u3153 import hedderich_chapter_9_unnumbered_3153


def test_hedderich9u3153_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3153(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u3153_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3153(x)
    assert isinstance(result, dict)
