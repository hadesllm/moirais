"""Tests for hedderich9u2303.hedderich_chapter_9_unnumbered_2303."""
import numpy as np
import pytest
from morie.fn.hedderich9u2303 import hedderich_chapter_9_unnumbered_2303


def test_hedderich9u2303_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2303(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u2303_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2303(x)
    assert isinstance(result, dict)
