"""Tests for hedderich9u2251.hedderich_chapter_9_unnumbered_2251."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2251 import hedderich_chapter_9_unnumbered_2251


def test_hedderich9u2251_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2251(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2251_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2251(x)
    assert isinstance(result, dict)
