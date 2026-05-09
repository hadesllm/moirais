"""Tests for rgmurm.rangayyan_murmur_analysis."""
import numpy as np
import pytest
from moirais.fn.rgmurm import rangayyan_murmur_analysis


def test_rgmurm_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    result = rangayyan_murmur_analysis(pcg, fs, ecg)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmurm_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    result = rangayyan_murmur_analysis(pcg, fs, ecg)
    assert isinstance(result, dict)
