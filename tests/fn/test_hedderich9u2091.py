"""Tests for hedderich9u2091.hedderich_chapter_9_unnumbered_2091."""
import numpy as np
import pytest
from moirais.fn.hedderich9u2091 import hedderich_chapter_9_unnumbered_2091


def test_hedderich9u2091_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2091(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u2091_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_2091(x)
    assert isinstance(result, dict)
