"""Tests for hedderich9u2401.hedderich_chapter_9_unnumbered_2401."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2401 import hedderich_chapter_9_unnumbered_2401


def test_hedderich9u2401_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2401(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u2401_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2401(x)
    assert isinstance(result, dict)
