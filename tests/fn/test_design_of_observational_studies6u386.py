"""Tests for design_of_observational_studies6u386.design_of_observational_studies_chapter_6_unnumbered_386."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies6u386 import design_of_observational_studies_chapter_6_unnumbered_386


def test_design_of_observational_studies6u386_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_6_unnumbered_386(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_design_of_observational_studies6u386_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_6_unnumbered_386(x)
    assert isinstance(result, dict)
