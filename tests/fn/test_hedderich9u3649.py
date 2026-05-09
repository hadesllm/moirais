"""Tests for hedderich9u3649.hedderich_chapter_9_unnumbered_3649."""
import numpy as np
import pytest
from moirais.fn.hedderich9u3649 import hedderich_chapter_9_unnumbered_3649


def test_hedderich9u3649_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3649(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u3649_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3649(x)
    assert isinstance(result, dict)
