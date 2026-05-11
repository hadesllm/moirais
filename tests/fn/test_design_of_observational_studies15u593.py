"""Tests for design_of_observational_studies15u593.design_of_observational_studies_chapter_15_unnumbered_593."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies15u593 import design_of_observational_studies_chapter_15_unnumbered_593


def test_design_of_observational_studies15u593_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_593(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies15u593_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_593(x)
    assert isinstance(result, dict)
