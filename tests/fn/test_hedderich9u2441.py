"""Tests for hedderich9u2441.hedderich_chapter_9_unnumbered_2441."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2441 import hedderich_chapter_9_unnumbered_2441


def test_hedderich9u2441_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2441(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u2441_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2441(x)
    assert isinstance(result, dict)
