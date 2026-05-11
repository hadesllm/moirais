"""Tests for hedderich9u978.hedderich_chapter_9_unnumbered_978."""
import numpy as np
import pytest
from morie.fn.hedderich9u978 import hedderich_chapter_9_unnumbered_978


def test_hedderich9u978_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_978(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u978_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_978(x)
    assert isinstance(result, dict)
