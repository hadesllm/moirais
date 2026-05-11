"""He who has a why to live can bear almost any how. — Friedrich Nietzsche"""
import numpy as np
import pytest
from morie.fn.christopher_gandrud_author_reproducible_research_with_r_and_4u21 import christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_21


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_21(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_21(x)
    assert isinstance(result, dict)
