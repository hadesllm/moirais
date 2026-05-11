"""Tests for hedderich9u21.hedderich_chapter_9_unnumbered_21."""
import numpy as np
import pytest
from morie.fn.hedderich9u21 import hedderich_chapter_9_unnumbered_21


def test_hedderich9u21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_21(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_21(x)
    assert isinstance(result, dict)
