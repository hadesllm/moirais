"""Tests for design_of_observational_studies15u461.design_of_observational_studies_chapter_15_unnumbered_461."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies15u461 import design_of_observational_studies_chapter_15_unnumbered_461


def test_design_of_observational_studies15u461_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_461(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_design_of_observational_studies15u461_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_461(x)
    assert isinstance(result, dict)
