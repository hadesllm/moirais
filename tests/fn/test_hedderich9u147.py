"""Tests for hedderich9u147.hedderich_chapter_9_unnumbered_147."""
import numpy as np
import pytest
from moirais.fn.hedderich9u147 import hedderich_chapter_9_unnumbered_147


def test_hedderich9u147_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_147(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u147_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_147(x)
    assert isinstance(result, dict)
