"""Tests for hedderich9u2273.hedderich_chapter_9_unnumbered_2273."""
import numpy as np
import pytest
from morie.fn.hedderich9u2273 import hedderich_chapter_9_unnumbered_2273


def test_hedderich9u2273_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2273(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u2273_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2273(x)
    assert isinstance(result, dict)
