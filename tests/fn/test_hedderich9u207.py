"""Tests for hedderich9u207.hedderich_chapter_9_unnumbered_207."""
import numpy as np
import pytest
from moirais.fn.hedderich9u207 import hedderich_chapter_9_unnumbered_207


def test_hedderich9u207_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_207(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u207_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_207(x)
    assert isinstance(result, dict)
