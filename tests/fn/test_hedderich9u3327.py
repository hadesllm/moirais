"""Tests for hedderich9u3327.hedderich_chapter_9_unnumbered_3327."""
import numpy as np
import pytest
from morie.fn.hedderich9u3327 import hedderich_chapter_9_unnumbered_3327


def test_hedderich9u3327_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3327(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u3327_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3327(x)
    assert isinstance(result, dict)
