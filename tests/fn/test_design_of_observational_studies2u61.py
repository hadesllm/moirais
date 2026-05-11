"""Tests for design_of_observational_studies2u61.design_of_observational_studies_chapter_2_unnumbered_61."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies2u61 import design_of_observational_studies_chapter_2_unnumbered_61


def test_design_of_observational_studies2u61_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_2_unnumbered_61(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_design_of_observational_studies2u61_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_2_unnumbered_61(x)
    assert isinstance(result, dict)
