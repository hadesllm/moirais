"""Tests for morie.fn.lsbemb -- LSB steganography."""

import numpy as np
from morie.fn.lsbemb import lsb_embed, lsb_extract, lsbemb
from morie.fn._containers import DescriptiveResult


class TestLsbemb:
    def test_alias(self):
        assert lsbemb is lsb_embed

    def test_roundtrip(self):
        cover = np.random.default_rng(42).integers(0, 256, 100)
        msg = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        r = lsb_embed(cover, msg)
        assert isinstance(r, DescriptiveResult)
        extracted = lsb_extract(r.value, len(msg))
        np.testing.assert_array_equal(extracted, msg)

    def test_psnr(self):
        cover = np.full(1000, 128, dtype=int)
        msg = np.array([1, 0, 1, 0])
        r = lsb_embed(cover, msg)
        assert r.extra["psnr"] > 30
