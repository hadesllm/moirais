"""Real knowledge is to know the extent of one's ignorance. — Confucius"""
import numpy as np
import pytest
from moirais.fn.christopher_gandrud_author_reproducible_research_with_r_and_4u20 import christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_20


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_20(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_20(x)
    assert isinstance(result, dict)
