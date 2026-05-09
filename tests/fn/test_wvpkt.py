"""Tests for wvpkt.py - Wavelet packet decomposition."""
import numpy as np
from moirais.fn.wvpkt import wavelet_packets, wvpkt


def test_wvpkt_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(128)
    result = wavelet_packets(x, level=2)
    assert result.name == "wavelet_packets"
    assert "tree" in result.extra


def test_wvpkt_leaf_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = wavelet_packets(x, level=3)
    assert result.extra["n_nodes"] == 8


def test_wvpkt_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = wvpkt(x, level=1)
    assert result.name == "wavelet_packets"
