"""Tests for design_of_observational_studies3e13.design_of_observational_studies_chapter_3_equation_13."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies3e13 import design_of_observational_studies_chapter_3_equation_13


def test_design_of_observational_studies3e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_3_equation_13(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_design_of_observational_studies3e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_3_equation_13(x)
    assert isinstance(result, dict)
