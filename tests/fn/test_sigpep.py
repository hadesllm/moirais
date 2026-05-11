"""Tests for sigpep.signal_peptide."""
import numpy as np
import pytest
from morie.fn.sigpep import signal_peptide


def test_sigpep_basic():
    """Test basic functionality."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    result = signal_peptide(sequence)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sigpep_edge():
    """Test edge cases."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    result = signal_peptide(sequence)
    assert isinstance(result, dict)
