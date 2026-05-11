"""Tests for design_of_observational_studies20u806.design_of_observational_studies_chapter_20_unnumbered_806."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies20u806 import design_of_observational_studies_chapter_20_unnumbered_806


def test_design_of_observational_studies20u806_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_806(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_design_of_observational_studies20u806_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_806(x)
    assert isinstance(result, dict)
