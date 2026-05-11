"""Tests for hedderich9u3602.hedderich_chapter_9_unnumbered_3602."""
import numpy as np
import pytest
from morie.fn.hedderich9u3602 import hedderich_chapter_9_unnumbered_3602


def test_hedderich9u3602_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3602(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u3602_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3602(x)
    assert isinstance(result, dict)
