"""Tests for moirais.fn.jllem — JL lemma dimension bound."""

import numpy as np
import pytest

from moirais.fn.jllem import jl_lemma_bound


class TestJlLemmaBound:

    def test_basic(self):
        res = jl_lemma_bound(1000, eps=0.1)
        assert res.value > 0
        assert res.extra["n"] == 1000

    def test_dimension_increases_with_n(self):
        d100 = jl_lemma_bound(100, eps=0.1).value
        d10000 = jl_lemma_bound(10000, eps=0.1).value
        assert d10000 > d100

    def test_dimension_increases_with_smaller_eps(self):
        d_loose = jl_lemma_bound(1000, eps=0.5).value
        d_tight = jl_lemma_bound(1000, eps=0.1).value
        assert d_tight > d_loose

    def test_invalid_eps(self):
        with pytest.raises(ValueError):
            jl_lemma_bound(100, eps=0)
        with pytest.raises(ValueError):
            jl_lemma_bound(100, eps=1.0)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            jl_lemma_bound(1, eps=0.1)
