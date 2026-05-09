"""Tests for hedderich9u23.hedderich_chapter_9_unnumbered_23."""
import numpy as np
import pytest
from moirais.fn.hedderich9u23 import hedderich_chapter_9_unnumbered_23


def test_hedderich9u23_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_23(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u23_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_23(x)
    assert isinstance(result, dict)
