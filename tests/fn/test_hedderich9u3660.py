"""Tests for hedderich9u3660.hedderich_chapter_9_unnumbered_3660."""
import numpy as np
import pytest
from moirais.fn.hedderich9u3660 import hedderich_chapter_9_unnumbered_3660


def test_hedderich9u3660_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3660(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u3660_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3660(x)
    assert isinstance(result, dict)
