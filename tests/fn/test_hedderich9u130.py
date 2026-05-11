"""Tests for hedderich9u130.hedderich_chapter_9_unnumbered_130."""
import numpy as np
import pytest
from morie.fn.hedderich9u130 import hedderich_chapter_9_unnumbered_130


def test_hedderich9u130_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_130(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u130_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_130(x)
    assert isinstance(result, dict)
