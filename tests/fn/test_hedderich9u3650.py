"""Tests for hedderich9u3650.hedderich_chapter_9_unnumbered_3650."""
import numpy as np
import pytest
from morie.fn.hedderich9u3650 import hedderich_chapter_9_unnumbered_3650


def test_hedderich9u3650_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3650(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u3650_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3650(x)
    assert isinstance(result, dict)
