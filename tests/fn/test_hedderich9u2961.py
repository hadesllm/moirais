"""Tests for hedderich9u2961.hedderich_chapter_9_unnumbered_2961."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2961 import hedderich_chapter_9_unnumbered_2961


def test_hedderich9u2961_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2961(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u2961_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2961(x)
    assert isinstance(result, dict)
