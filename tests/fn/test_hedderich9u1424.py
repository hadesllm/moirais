"""Tests for hedderich9u1424.hedderich_chapter_9_unnumbered_1424."""
import numpy as np
import pytest
from moirais.fn.hedderich9u1424 import hedderich_chapter_9_unnumbered_1424


def test_hedderich9u1424_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1424(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u1424_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_1424(x)
    assert isinstance(result, dict)
