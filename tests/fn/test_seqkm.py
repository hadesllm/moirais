"""Tests for moirais.fn.seqkm."""
import numpy as np
from moirais.fn.seqkm import kmer_frequency


def test_seqkm_smoke():
    rng = np.random.default_rng(42)
    result = kmer_frequency(sequence="ACGTACGT")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.seqkm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
