"""Test mel_cepstral_coeffs (mfcc)."""
import numpy as np
from morie.fn.mfcc import mel_cepstral_coeffs, mfcc
from morie.fn._containers import DescriptiveResult


class TestMfcc:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(512)
        result = mel_cepstral_coeffs(x, fs=16000.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "mel_cepstral_coeffs"

    def test_n_coeffs(self):
        x = np.random.default_rng(42).standard_normal(512)
        result = mel_cepstral_coeffs(x, n_mfcc=13)
        assert len(result.extra["mfcc"]) == 13

    def test_alias(self):
        assert mfcc is mel_cepstral_coeffs
