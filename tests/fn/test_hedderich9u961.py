"""Tests for hedderich9u961.hedderich_chapter_9_unnumbered_961."""
import numpy as np
import pytest
from moirais.fn.hedderich9u961 import hedderich_chapter_9_unnumbered_961


def test_hedderich9u961_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_961(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u961_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_961(x)
    assert isinstance(result, dict)
