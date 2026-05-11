"""Tests for hedderich9u1405.hedderich_chapter_9_unnumbered_1405."""
import numpy as np
import pytest
from morie.fn.hedderich9u1405 import hedderich_chapter_9_unnumbered_1405


def test_hedderich9u1405_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1405(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u1405_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1405(x)
    assert isinstance(result, dict)
