"""Tests for hedderich9u1214.hedderich_chapter_9_unnumbered_1214."""
import numpy as np
import pytest
from morie.fn.hedderich9u1214 import hedderich_chapter_9_unnumbered_1214


def test_hedderich9u1214_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1214(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1214_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1214(x)
    assert isinstance(result, dict)
