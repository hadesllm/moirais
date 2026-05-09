"""Tests for hedderich9u420.hedderich_chapter_9_unnumbered_420."""
import numpy as np
import pytest
from moirais.fn.hedderich9u420 import hedderich_chapter_9_unnumbered_420


def test_hedderich9u420_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_420(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u420_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_420(x)
    assert isinstance(result, dict)
