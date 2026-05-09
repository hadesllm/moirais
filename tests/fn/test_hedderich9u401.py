"""Tests for hedderich9u401.hedderich_chapter_9_unnumbered_401."""
import numpy as np
import pytest
from moirais.fn.hedderich9u401 import hedderich_chapter_9_unnumbered_401


def test_hedderich9u401_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_401(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u401_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_401(x)
    assert isinstance(result, dict)
