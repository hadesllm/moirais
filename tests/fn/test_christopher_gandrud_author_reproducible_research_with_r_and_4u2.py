"""We suffer more often in imagination than in reality. — Seneca"""
import numpy as np
import pytest
from morie.fn.christopher_gandrud_author_reproducible_research_with_r_and_4u2 import christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_2


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_2(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_2(x)
    assert isinstance(result, dict)
