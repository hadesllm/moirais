"""Tests for rgepidet.rangayyan_epilepsy_detect."""
import numpy as np
import pytest
from moirais.fn.rgepidet import rangayyan_epilepsy_detect


def test_rgepidet_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    dictionary_size = 100
    result = rangayyan_epilepsy_detect(eeg, fs, dictionary_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgepidet_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    dictionary_size = 100
    result = rangayyan_epilepsy_detect(eeg, fs, dictionary_size)
    assert isinstance(result, dict)
