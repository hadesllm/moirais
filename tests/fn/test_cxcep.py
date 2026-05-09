"""Test complex_cepstrum (cxcep)."""
import numpy as np
import pytest

from moirais.fn.cxcep import complex_cepstrum, cxcep
from moirais.fn._containers import DescriptiveResult


class TestComplexCepstrum:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = complex_cepstrum(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "complex_cepstrum"

    def test_extra_keys(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = complex_cepstrum(x)
        assert "cepstrum" in result.extra
        assert "quefrency" in result.extra

    def test_output_lengths(self):
        x = np.random.default_rng(42).standard_normal(128)
        result = complex_cepstrum(x)
        assert len(result.extra["cepstrum"]) == 128
        assert len(result.extra["quefrency"]) == 128

    def test_alias(self):
        assert cxcep is complex_cepstrum
