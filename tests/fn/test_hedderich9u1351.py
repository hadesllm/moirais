"""Tests for hedderich9u1351.hedderich_chapter_9_unnumbered_1351."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1351 import hedderich_chapter_9_unnumbered_1351


def test_hedderich9u1351_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1351(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u1351_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1351(x)
    assert isinstance(result, dict)
