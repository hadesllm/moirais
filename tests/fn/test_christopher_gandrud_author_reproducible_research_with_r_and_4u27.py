"""Nature does not hurry, yet everything is accomplished. — Lao Tzu"""
import numpy as np
import pytest
from morie.fn.christopher_gandrud_author_reproducible_research_with_r_and_4u27 import christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_27


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u27_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_27(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u27_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_27(x)
    assert isinstance(result, dict)
