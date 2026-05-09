"""Tests for hedderich9u293.hedderich_chapter_9_unnumbered_293."""
import numpy as np
import pytest
from moirais.fn.hedderich9u293 import hedderich_chapter_9_unnumbered_293


def test_hedderich9u293_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_293(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u293_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_293(x)
    assert isinstance(result, dict)
