"""Tests for fbank.py - Filter bank design."""
import numpy as np
from moirais.fn.fbank import filter_bank_design, fbank


def test_fbank_returns_descriptive_result():
    result = filter_bank_design(wavelet="db4")
    assert result.name == "filter_bank_design"
    assert "lo_d" in result.extra
    assert "hi_d" in result.extra


def test_fbank_reconstruction_filters():
    result = filter_bank_design(wavelet="haar")
    lo_d = result.extra["lo_d"]
    lo_r = result.extra["lo_r"]
    np.testing.assert_array_equal(lo_r, lo_d[::-1])


def test_fbank_alias():
    result = fbank(wavelet="haar")
    assert result.name == "filter_bank_design"
