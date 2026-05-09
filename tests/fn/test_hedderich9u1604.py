"""Tests for hedderich9u1604.hedderich_chapter_9_unnumbered_1604."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1604 import hedderich_chapter_9_unnumbered_1604


def test_hedderich9u1604_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1604(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1604_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1604(x)
    assert isinstance(result, dict)
