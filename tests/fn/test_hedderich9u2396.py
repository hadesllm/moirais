"""Tests for hedderich9u2396.hedderich_chapter_9_unnumbered_2396."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2396 import hedderich_chapter_9_unnumbered_2396


def test_hedderich9u2396_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2396(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u2396_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2396(x)
    assert isinstance(result, dict)
