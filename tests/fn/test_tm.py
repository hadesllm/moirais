"""Tests for morie.fn.tm — melting temperature."""
import pytest
from morie.fn.tm import melting_temperature


class TestMeltingTemp:
    def test_short_oligo(self):
        res = melting_temperature("ATCGATCG")
        assert 10 < res.estimate < 80

    def test_gc_rich_higher(self):
        gc_rich = melting_temperature("GCGCGCGCGC")
        at_rich = melting_temperature("ATATATATAT")
        assert gc_rich.estimate > at_rich.estimate

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            melting_temperature("")
