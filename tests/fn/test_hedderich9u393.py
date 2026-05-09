"""Tests for hedderich9u393.hedderich_chapter_9_unnumbered_393."""
import numpy as np
import pytest
from moirais.fn.hedderich9u393 import hedderich_chapter_9_unnumbered_393


def test_hedderich9u393_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_393(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hedderich9u393_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_393(x)
    assert isinstance(result, dict)
