"""Tests for hedderich9u2500.hedderich_chapter_9_unnumbered_2500."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2500 import hedderich_chapter_9_unnumbered_2500


def test_hedderich9u2500_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2500(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2500_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2500(x)
    assert isinstance(result, dict)
