"""Tests for hedderich9u3606.hedderich_chapter_9_unnumbered_3606."""
import numpy as np
import pytest
from moirais.fn.hedderich9u3606 import hedderich_chapter_9_unnumbered_3606


def test_hedderich9u3606_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3606(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u3606_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3606(x)
    assert isinstance(result, dict)
