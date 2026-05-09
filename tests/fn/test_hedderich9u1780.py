"""Tests for hedderich9u1780.hedderich_chapter_9_unnumbered_1780."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1780 import hedderich_chapter_9_unnumbered_1780


def test_hedderich9u1780_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1780(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1780_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1780(x)
    assert isinstance(result, dict)
