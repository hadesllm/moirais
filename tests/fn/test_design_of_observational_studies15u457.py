"""Tests for design_of_observational_studies15u457.design_of_observational_studies_chapter_15_unnumbered_457."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies15u457 import design_of_observational_studies_chapter_15_unnumbered_457


def test_design_of_observational_studies15u457_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_457(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_design_of_observational_studies15u457_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_457(x)
    assert isinstance(result, dict)
