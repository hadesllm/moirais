"""Tests for moirais.fn.vq — vector quantization."""

import numpy as np
import pytest

from moirais.fn.vq import vector_quantize


class TestVectorQuantize:

    def test_returns_result(self):
        codebook = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        x = np.array([[0.1, 0.1], [0.9, 0.1]])
        res = vector_quantize(x, codebook)
        assert res.name == "vector_quantize"

    def test_nearest_codeword(self):
        codebook = np.array([[0.0], [1.0], [2.0]])
        x = np.array([[0.1], [1.9]])
        res = vector_quantize(x, codebook)
        assert list(res.extra["codes"]) == [0, 2]

    def test_1d_input(self):
        codebook = np.array([[0.0], [1.0]])
        x = np.array([0.3])
        res = vector_quantize(x, codebook)
        assert res.extra["codes"][0] == 0
