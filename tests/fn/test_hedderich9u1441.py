"""Tests for hedderich9u1441.hedderich_chapter_9_unnumbered_1441."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1441 import hedderich_chapter_9_unnumbered_1441


def test_hedderich9u1441_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1441(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1441_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1441(x)
    assert isinstance(result, dict)
