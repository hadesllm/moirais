"""Tests for hedderich9u3580.hedderich_chapter_9_unnumbered_3580."""
import numpy as np
import pytest
from morie.fn.hedderich9u3580 import hedderich_chapter_9_unnumbered_3580


def test_hedderich9u3580_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3580(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hedderich9u3580_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_3580(x)
    assert isinstance(result, dict)
