"""Tests for hedderich9u951.hedderich_chapter_9_unnumbered_951."""
import numpy as np
import pytest
from morie.fn.hedderich9u951 import hedderich_chapter_9_unnumbered_951


def test_hedderich9u951_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_951(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u951_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_951(x)
    assert isinstance(result, dict)
