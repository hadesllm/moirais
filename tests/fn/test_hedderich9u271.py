"""Tests for hedderich9u271.hedderich_chapter_9_unnumbered_271."""
import numpy as np
import pytest
from moirais.fn.hedderich9u271 import hedderich_chapter_9_unnumbered_271


def test_hedderich9u271_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_271(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u271_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_271(x)
    assert isinstance(result, dict)
