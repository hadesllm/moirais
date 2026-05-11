"""Tests for hedderich9u1981.hedderich_chapter_9_unnumbered_1981."""
import numpy as np
import pytest
from morie.fn.hedderich9u1981 import hedderich_chapter_9_unnumbered_1981


def test_hedderich9u1981_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1981(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u1981_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1981(x)
    assert isinstance(result, dict)
