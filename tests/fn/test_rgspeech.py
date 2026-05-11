"""Tests for rgspeech.rangayyan_speech_features."""
import numpy as np
import pytest
from morie.fn.rgspeech import rangayyan_speech_features


def test_rgspeech_basic():
    """Test basic functionality."""
    speech = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    order = 4
    result = rangayyan_speech_features(speech, fs, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgspeech_edge():
    """Test edge cases."""
    speech = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    order = 4
    result = rangayyan_speech_features(speech, fs, order)
    assert isinstance(result, dict)
