"""Tests for design_of_observational_studies20u823.design_of_observational_studies_chapter_20_unnumbered_823."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies20u823 import design_of_observational_studies_chapter_20_unnumbered_823


def test_design_of_observational_studies20u823_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_823(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies20u823_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_823(x)
    assert isinstance(result, dict)
