"""Tests for hhtfn.py - Hilbert-Huang Transform."""
import numpy as np
from moirais.fn.hhtfn import hilbert_huang, hhtfn


def test_hhtfn_returns_descriptive_result():
    t = np.linspace(0, 1, 256)
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 30 * t)
    result = hilbert_huang(x, fs=256.0)
    assert result.name == "hilbert_huang"
    assert "imfs" in result.extra


def test_hhtfn_extracts_imfs():
    t = np.linspace(0, 1, 256)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    result = hilbert_huang(x, fs=256.0)
    assert len(result.extra["imfs"]) >= 1


def test_hhtfn_alias():
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    result = hhtfn(x)
    assert result.name == "hilbert_huang"
