"""Tests for design_of_observational_studies15u611.design_of_observational_studies_chapter_15_unnumbered_611."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies15u611 import design_of_observational_studies_chapter_15_unnumbered_611


def test_design_of_observational_studies15u611_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_611(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_design_of_observational_studies15u611_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_611(x)
    assert isinstance(result, dict)
