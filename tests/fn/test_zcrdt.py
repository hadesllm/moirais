"""Tests for moirais.fn.zcrdt — zero-crossing rate detection."""
import numpy as np
import pytest

from moirais.fn.zcrdt import zcr_detect, zcrdt


def test_sine_zcr():
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 50 * t)
    result = zcr_detect(x, frame_len=100, hop=50)
    assert result.value > 0
    assert result.extra["n_frames"] > 0


def test_constant_signal():
    x = np.ones(500)
    result = zcr_detect(x, frame_len=100, hop=50)
    assert result.value == 0.0


def test_positions_length():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(512)
    result = zcr_detect(x, frame_len=256, hop=128)
    assert len(result.extra["positions"]) == result.extra["n_frames"]


def test_alias():
    assert zcrdt is zcr_detect
