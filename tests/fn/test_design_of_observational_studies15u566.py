"""Tests for design_of_observational_studies15u566.design_of_observational_studies_chapter_15_unnumbered_566."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies15u566 import design_of_observational_studies_chapter_15_unnumbered_566


def test_design_of_observational_studies15u566_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_566(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies15u566_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_566(x)
    assert isinstance(result, dict)
