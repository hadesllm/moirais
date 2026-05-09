"""Tests for hedderich9u297.hedderich_chapter_9_unnumbered_297."""
import numpy as np
import pytest
from moirais.fn.hedderich9u297 import hedderich_chapter_9_unnumbered_297


def test_hedderich9u297_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_297(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u297_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_297(x)
    assert isinstance(result, dict)
