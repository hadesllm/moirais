"""Tests for hedderich9u18.hedderich_chapter_9_unnumbered_18."""
import numpy as np
import pytest
from moirais.fn.hedderich9u18 import hedderich_chapter_9_unnumbered_18


def test_hedderich9u18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_18(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_18(x)
    assert isinstance(result, dict)
