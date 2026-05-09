"""Tests for design_of_observational_studies6u390.design_of_observational_studies_chapter_6_unnumbered_390."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies6u390 import design_of_observational_studies_chapter_6_unnumbered_390


def test_design_of_observational_studies6u390_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_6_unnumbered_390(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_design_of_observational_studies6u390_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_6_unnumbered_390(x)
    assert isinstance(result, dict)
