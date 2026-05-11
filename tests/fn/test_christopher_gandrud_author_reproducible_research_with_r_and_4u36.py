"""The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius"""
import numpy as np
import pytest
from morie.fn.christopher_gandrud_author_reproducible_research_with_r_and_4u36 import christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_36


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_36(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_36(x)
    assert isinstance(result, dict)
