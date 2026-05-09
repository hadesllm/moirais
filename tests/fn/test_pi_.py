"""Tests for moirais.fn.pi_ — nucleotide diversity."""
import pytest
from moirais.fn.pi_ import nucleotide_diversity


class TestNucleotideDiversity:
    def test_identical_seqs(self):
        res = nucleotide_diversity(["ATCG", "ATCG", "ATCG"])
        assert res.statistic == pytest.approx(0.0)

    def test_different_seqs(self):
        res = nucleotide_diversity(["AAAA", "TTTT"])
        assert res.statistic == pytest.approx(1.0)

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            nucleotide_diversity(["ATCG"])
