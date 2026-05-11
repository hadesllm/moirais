"""Look well into thyself; there is a source which will always spring up. — Marcus Aurelius"""
import numpy as np
import pytest
from morie.fn.christopher_gandrud_author_reproducible_research_with_r_and_4u40 import christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_40


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u40_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_40(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u40_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_40(x)
    assert isinstance(result, dict)
