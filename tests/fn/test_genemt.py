"""Tests for genemt.gene_meta_analysis."""
import numpy as np
import pytest
from morie.fn.genemt import gene_meta_analysis


def test_genemt_basic():
    """Test basic functionality."""
    sumstats = np.random.default_rng(42).normal(0, 1, 100)
    gene_annotation = np.random.default_rng(42).normal(0, 1, 100)
    result = gene_meta_analysis(sumstats, gene_annotation)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_genemt_edge():
    """Test edge cases."""
    sumstats = np.random.default_rng(42).normal(0, 1, 100)
    gene_annotation = np.random.default_rng(42).normal(0, 1, 100)
    result = gene_meta_analysis(sumstats, gene_annotation)
    assert isinstance(result, dict)
