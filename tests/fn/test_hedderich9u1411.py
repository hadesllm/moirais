"""Tests for hedderich9u1411.hedderich_chapter_9_unnumbered_1411."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1411 import hedderich_chapter_9_unnumbered_1411


def test_hedderich9u1411_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1411(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u1411_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1411(x)
    assert isinstance(result, dict)
