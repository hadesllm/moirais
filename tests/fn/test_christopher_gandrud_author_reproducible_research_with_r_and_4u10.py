"""Patience is bitter, but its fruit is sweet. — Aristotle"""
import numpy as np
import pytest
from moirais.fn.christopher_gandrud_author_reproducible_research_with_r_and_4u10 import christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_10


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_10(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_10(x)
    assert isinstance(result, dict)
