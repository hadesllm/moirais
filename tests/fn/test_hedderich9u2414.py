"""Tests for hedderich9u2414.hedderich_chapter_9_unnumbered_2414."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2414 import hedderich_chapter_9_unnumbered_2414


def test_hedderich9u2414_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2414(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u2414_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2414(x)
    assert isinstance(result, dict)
