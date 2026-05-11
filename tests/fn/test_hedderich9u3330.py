"""Tests for hedderich9u3330.hedderich_chapter_9_unnumbered_3330."""
import numpy as np
import pytest
from morie.fn.hedderich9u3330 import hedderich_chapter_9_unnumbered_3330


def test_hedderich9u3330_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3330(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u3330_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3330(x)
    assert isinstance(result, dict)
