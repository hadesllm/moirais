"""Tests for morie.fn.maf -- Minor allele frequency."""

import pytest
from morie.fn.maf import minor_allele_frequency


class TestMAF:
    def test_known_frequency(self):
        """[0,0,1,1,2] => p = 4/10 = 0.4, MAF = 0.4."""
        res = minor_allele_frequency([0, 0, 1, 1, 2])
        assert res.measure == "MAF"
        assert res.estimate == pytest.approx(0.4, abs=1e-10)

    def test_all_homozygous_ref(self):
        """All 0 => p = 0.0, MAF = 0.0."""
        res = minor_allele_frequency([0, 0, 0, 0])
        assert res.estimate == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        """MAF should be <= 0.5 regardless of which allele is more common."""
        res = minor_allele_frequency([2, 2, 2, 2, 1])
        assert res.estimate <= 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            minor_allele_frequency([])

    def test_invalid_genotype_raises(self):
        with pytest.raises(ValueError):
            minor_allele_frequency([0, 1, 3])
