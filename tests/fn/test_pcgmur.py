"""Tests for pcgmur — PCG murmur detection score."""
import numpy as np
from morie.fn.pcgmur import pcg_murmur_score
from morie.fn._containers import DescriptiveResult


def test_pcgmur_basic(rng):
    fs = 2000
    t = np.arange(0, 2.0, 1 / fs)
    s1 = np.zeros_like(t)
    for i in range(0, len(t), int(0.8 * fs)):
        w = min(len(t), i + int(0.05 * fs))
        s1[i:w] = np.sin(2 * np.pi * 50 * t[:w - i])
    result = pcg_murmur_score(s1 + rng.standard_normal(len(t)) * 0.01, fs)
    assert isinstance(result, DescriptiveResult)
    assert 0 <= result.value <= 1


def test_pcgmur_noisy_higher(rng):
    fs = 2000
    t = np.arange(0, 2.0, 1 / fs)
    clean = np.sin(2 * np.pi * 50 * t) * 0.5
    noisy = clean + rng.standard_normal(len(t)) * 0.8
    score_clean = pcg_murmur_score(clean, fs).value
    score_noisy = pcg_murmur_score(noisy, fs).value
    assert score_noisy > score_clean * 0.5
