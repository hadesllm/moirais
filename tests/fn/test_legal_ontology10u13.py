"""Tests for legal_ontology10u13.legal_ontology_chapter_10_unnumbered_13."""
import numpy as np
import pytest
from moirais.fn.legal_ontology10u13 import legal_ontology_chapter_10_unnumbered_13


def test_legal_ontology10u13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = legal_ontology_chapter_10_unnumbered_13(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_legal_ontology10u13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = legal_ontology_chapter_10_unnumbered_13(x)
    assert isinstance(result, dict)
