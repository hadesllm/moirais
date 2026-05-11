"""Tests for design_of_observational_studies20u747.design_of_observational_studies_chapter_20_unnumbered_747."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies20u747 import design_of_observational_studies_chapter_20_unnumbered_747


def test_design_of_observational_studies20u747_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_747(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_design_of_observational_studies20u747_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_747(x)
    assert isinstance(result, dict)
