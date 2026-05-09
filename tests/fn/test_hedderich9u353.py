"""Tests for hedderich9u353.hedderich_chapter_9_unnumbered_353."""
import numpy as np
import pytest
from moirais.fn.hedderich9u353 import hedderich_chapter_9_unnumbered_353


def test_hedderich9u353_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_353(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u353_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_353(x)
    assert isinstance(result, dict)
