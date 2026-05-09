"""He who would learn to fly one day must first learn to stand and walk. — Friedrich Nietzsche"""
import numpy as np
import pytest
from moirais.fn.christopher_gandrud_author_reproducible_research_with_r_and_4u16 import christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_16


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_16(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_christopher_gandrud_author_reproducible_research_with_r_and_4u16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_16(x)
    assert isinstance(result, dict)
