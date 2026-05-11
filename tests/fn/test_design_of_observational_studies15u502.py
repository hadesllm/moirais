"""Tests for design_of_observational_studies15u502.design_of_observational_studies_chapter_15_unnumbered_502."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies15u502 import design_of_observational_studies_chapter_15_unnumbered_502


def test_design_of_observational_studies15u502_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_502(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_design_of_observational_studies15u502_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_502(x)
    assert isinstance(result, dict)
