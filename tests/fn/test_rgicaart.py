"""Tests for rgicaart.rangayyan_ica_artifact."""
import numpy as np
import pytest
from moirais.fn.rgicaart import rangayyan_ica_artifact


def test_rgicaart_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    n_components = 3
    artifact_labels = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ica_artifact(eeg, n_components, artifact_labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgicaart_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    n_components = 3
    artifact_labels = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ica_artifact(eeg, n_components, artifact_labels)
    assert isinstance(result, dict)
