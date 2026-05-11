"""Tests for morie.fn.pqdec — PolarQuant decoder."""

import numpy as np
import pytest

from morie.fn.pqenc import polarquant_encode
from morie.fn.pqdec import polarquant_decode


class TestPolarquantDecode:

    def test_roundtrip(self):
        x = np.array([3.0, 4.0])
        enc = polarquant_encode(x)
        dec = polarquant_decode(enc.extra["magnitude"], enc.extra["direction"])
        assert np.allclose(dec.extra["reconstructed"], x, atol=1e-10)

    def test_magnitude_preserved(self):
        x = np.random.default_rng(42).standard_normal(32)
        enc = polarquant_encode(x)
        dec = polarquant_decode(enc.extra["magnitude"], enc.extra["direction"])
        assert dec.value == pytest.approx(np.linalg.norm(x), abs=1e-10)
