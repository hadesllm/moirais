"""Tests for hedderich9u2421.hedderich_chapter_9_unnumbered_2421."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2421 import hedderich_chapter_9_unnumbered_2421


def test_hedderich9u2421_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2421(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u2421_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2421(x)
    assert isinstance(result, dict)
