"""Tests for moirais.fn.adpthr — adaptive threshold detection."""
import numpy as np
import pytest

from moirais.fn.adpthr import adaptive_threshold_detect, adpthr


def test_no_detections_in_noise():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = adaptive_threshold_detect(x, window=50, k=5.0)
    assert result.extra["window"] == 50


def test_spike_detected():
    x = np.zeros(100)
    x[50] = 10.0
    result = adaptive_threshold_detect(x, window=20, k=1.5)
    assert 50 in result.extra["detections"]


def test_threshold_shape():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = adaptive_threshold_detect(x, window=10, k=2.0)
    assert len(result.extra["threshold"]) == 100


def test_alias():
    assert adpthr is adaptive_threshold_detect
