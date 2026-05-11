"""Tests for design_of_observational_studies6u366.design_of_observational_studies_chapter_6_unnumbered_366."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies6u366 import design_of_observational_studies_chapter_6_unnumbered_366


def test_design_of_observational_studies6u366_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_6_unnumbered_366(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies6u366_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_6_unnumbered_366(x)
    assert isinstance(result, dict)
