"""Tests for alftpl.alphafold_template_embed."""
import numpy as np
import pytest
from moirais.fn.alftpl import alphafold_template_embed


def test_alftpl_basic():
    """Test basic functionality."""
    templates = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = alphafold_template_embed(templates, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alftpl_edge():
    """Test edge cases."""
    templates = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = alphafold_template_embed(templates, z)
    assert isinstance(result, dict)
