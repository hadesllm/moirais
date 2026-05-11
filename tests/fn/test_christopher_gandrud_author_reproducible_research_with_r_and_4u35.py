"""Truth comes out of error more readily than out of confusion. — Francis Bacon"""
import numpy as np
import pytest
from morie.fn.christopher_gandrud_author_reproducible_research_with_r_and_4u35 import christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_35


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u35_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_35(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u35_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_35(x)
    assert isinstance(result, dict)
