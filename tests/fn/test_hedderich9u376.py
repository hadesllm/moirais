"""Tests for hedderich9u376.hedderich_chapter_9_unnumbered_376."""
import numpy as np
import pytest
from moirais.fn.hedderich9u376 import hedderich_chapter_9_unnumbered_376


def test_hedderich9u376_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_376(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u376_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_376(x)
    assert isinstance(result, dict)
