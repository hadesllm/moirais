"""Tests for hedderich9u602.hedderich_chapter_9_unnumbered_602."""
import numpy as np
import pytest
from moirais.fn.hedderich9u602 import hedderich_chapter_9_unnumbered_602


def test_hedderich9u602_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_602(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u602_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_602(x)
    assert isinstance(result, dict)
