"""Tests for hedderich9u80.hedderich_chapter_9_unnumbered_80."""
import numpy as np
import pytest
from morie.fn.hedderich9u80 import hedderich_chapter_9_unnumbered_80


def test_hedderich9u80_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_80(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u80_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_80(x)
    assert isinstance(result, dict)
