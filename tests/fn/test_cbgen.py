"""Tests for morie.fn.cbgen — codebook generation."""

import numpy as np
import pytest

from morie.fn.cbgen import codebook_generate


class TestCodebookGenerate:

    def test_returns_result(self):
        data = np.random.default_rng(42).standard_normal((100, 4))
        res = codebook_generate(data, k=8)
        assert res.name == "codebook_generate"

    def test_codebook_shape(self):
        data = np.random.default_rng(0).standard_normal((100, 4))
        res = codebook_generate(data, k=8)
        assert res.extra["codebook"].shape == (8, 4)

    def test_labels_valid(self):
        data = np.random.default_rng(1).standard_normal((50, 2))
        res = codebook_generate(data, k=4)
        assert np.all(res.extra["labels"] >= 0)
        assert np.all(res.extra["labels"] < 4)

    def test_1d_input(self):
        data = np.random.default_rng(2).standard_normal(100)
        res = codebook_generate(data, k=4)
        assert res.extra["codebook"].shape == (4, 1)
