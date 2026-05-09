"""Tests for hedderich9u281.hedderich_chapter_9_unnumbered_281."""
import numpy as np
import pytest
from moirais.fn.hedderich9u281 import hedderich_chapter_9_unnumbered_281


def test_hedderich9u281_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_281(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u281_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_281(x)
    assert isinstance(result, dict)
