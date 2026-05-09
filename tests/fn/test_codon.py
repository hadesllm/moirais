"""Tests for moirais.fn.codon — codon usage."""
import pytest
from moirais.fn.codon import codon_usage


class TestCodonUsage:
    def test_basic(self):
        res = codon_usage("ATGATGATG")
        assert res.extra["counts"]["ATG"] == 3
        assert res.extra["n_codons"] == 3

    def test_short_raises(self):
        with pytest.raises(ValueError):
            codon_usage("AT")
