"""Tests for chirg.py - Chirp generation."""
import numpy as np
from morie.fn.chirg import chirp_generate, chirg


def test_chirg_returns_descriptive_result():
    result = chirp_generate(f0=10, f1=100, T=1.0, fs=1000)
    assert result.name == "chirp_generate"
    assert "signal" in result.extra


def test_chirg_signal_length():
    result = chirp_generate(f0=1, f1=10, T=2.0, fs=100)
    assert len(result.extra["signal"]) == 200


def test_chirg_alias():
    result = chirg(f0=5, f1=50, T=0.5, fs=500)
    assert result.name == "chirp_generate"
