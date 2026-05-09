"""Tests for metabd.metagenome_binning."""
import numpy as np
import pytest
from moirais.fn.metabd import metagenome_binning


def test_metabd_basic():
    """Test basic functionality."""
    contigs = np.random.default_rng(42).normal(0, 1, 100)
    abundance = np.random.default_rng(42).normal(0, 1, 100)
    result = metagenome_binning(contigs, abundance)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_metabd_edge():
    """Test edge cases."""
    contigs = np.random.default_rng(42).normal(0, 1, 100)
    abundance = np.random.default_rng(42).normal(0, 1, 100)
    result = metagenome_binning(contigs, abundance)
    assert isinstance(result, dict)
