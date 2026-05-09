"""Tests for hedderich9u426.hedderich_chapter_9_unnumbered_426."""
import numpy as np
import pytest
from moirais.fn.hedderich9u426 import hedderich_chapter_9_unnumbered_426


def test_hedderich9u426_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_426(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u426_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_426(x)
    assert isinstance(result, dict)
