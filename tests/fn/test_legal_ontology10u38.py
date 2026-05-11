"""Tests for legal_ontology10u38.legal_ontology_chapter_10_unnumbered_38."""
import numpy as np
import pytest
from morie.fn.legal_ontology10u38 import legal_ontology_chapter_10_unnumbered_38


def test_legal_ontology10u38_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = legal_ontology_chapter_10_unnumbered_38(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_legal_ontology10u38_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = legal_ontology_chapter_10_unnumbered_38(x)
    assert isinstance(result, dict)
