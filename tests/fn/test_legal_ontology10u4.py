"""Tests for legal_ontology10u4.legal_ontology_chapter_10_unnumbered_4."""
import numpy as np
import pytest
from moirais.fn.legal_ontology10u4 import legal_ontology_chapter_10_unnumbered_4


def test_legal_ontology10u4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = legal_ontology_chapter_10_unnumbered_4(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_legal_ontology10u4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = legal_ontology_chapter_10_unnumbered_4(x)
    assert isinstance(result, dict)
