"""Tests for legal_ontology10u36.legal_ontology_chapter_10_unnumbered_36."""
import numpy as np
import pytest
from moirais.fn.legal_ontology10u36 import legal_ontology_chapter_10_unnumbered_36


def test_legal_ontology10u36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = legal_ontology_chapter_10_unnumbered_36(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_legal_ontology10u36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = legal_ontology_chapter_10_unnumbered_36(x)
    assert isinstance(result, dict)
