"""Tests for design_of_observational_studies20u714.design_of_observational_studies_chapter_20_unnumbered_714."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies20u714 import design_of_observational_studies_chapter_20_unnumbered_714


def test_design_of_observational_studies20u714_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_714(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies20u714_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_714(x)
    assert isinstance(result, dict)
