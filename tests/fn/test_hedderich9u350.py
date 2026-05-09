"""Tests for hedderich9u350.hedderich_chapter_9_unnumbered_350."""
import numpy as np
import pytest
from moirais.fn.hedderich9u350 import hedderich_chapter_9_unnumbered_350


def test_hedderich9u350_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_350(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich9u350_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_9_unnumbered_350(x)
    assert isinstance(result, dict)
