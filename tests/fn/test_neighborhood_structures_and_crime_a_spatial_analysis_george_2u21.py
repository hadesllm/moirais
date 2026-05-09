"""Tests for neighborhood_structures_and_crime_a_spatial_analysis_george_2u21.neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_21."""
import numpy as np
import pytest
from moirais.fn.neighborhood_structures_and_crime_a_spatial_analysis_george_2u21 import neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_21


def test_neighborhood_structures_and_crime_a_spatial_analysis_george_2u21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_21(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_neighborhood_structures_and_crime_a_spatial_analysis_george_2u21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_21(x)
    assert isinstance(result, dict)
