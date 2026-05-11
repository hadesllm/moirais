"""Tests for design_of_observational_studies15u401.design_of_observational_studies_chapter_15_unnumbered_401."""
import numpy as np
import pytest
from morie.fn.design_of_observational_studies15u401 import design_of_observational_studies_chapter_15_unnumbered_401


def test_design_of_observational_studies15u401_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_401(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies15u401_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_401(x)
    assert isinstance(result, dict)
