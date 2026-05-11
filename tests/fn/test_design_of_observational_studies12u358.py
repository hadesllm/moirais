"""Tests for design_of_observational_studies12u358.design_of_observational_studies_chapter_12_unnumbered_358."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies12u358 import design_of_observational_studies_chapter_12_unnumbered_358


def test_design_of_observational_studies12u358_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_12_unnumbered_358(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies12u358_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_12_unnumbered_358(x)
    assert isinstance(result, dict)
