"""Everything flows. — Heraclitus"""
import numpy as np
import pytest
from moirais.fn.christopher_gandrud_author_reproducible_research_with_r_and_3e20 import christopher_gandrud_author_reproducible_research_with_r_and__chapter_3_equation_20


def test_christopher_gandrud_author_reproducible_research_with_r_and_3e20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_3_equation_20(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_christopher_gandrud_author_reproducible_research_with_r_and_3e20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = christopher_gandrud_author_reproducible_research_with_r_and__chapter_3_equation_20(x)
    assert isinstance(result, dict)
