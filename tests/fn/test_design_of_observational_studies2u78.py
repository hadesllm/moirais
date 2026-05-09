"""Tests for design_of_observational_studies2u78.design_of_observational_studies_chapter_2_unnumbered_78."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies2u78 import design_of_observational_studies_chapter_2_unnumbered_78


def test_design_of_observational_studies2u78_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_2_unnumbered_78(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies2u78_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_2_unnumbered_78(x)
    assert isinstance(result, dict)
