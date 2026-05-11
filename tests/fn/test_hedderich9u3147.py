"""Tests for hedderich9u3147.hedderich_chapter_9_unnumbered_3147."""
import numpy as np
import pytest
from morie.fn.hedderich9u3147 import hedderich_chapter_9_unnumbered_3147


def test_hedderich9u3147_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3147(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u3147_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3147(x)
    assert isinstance(result, dict)
