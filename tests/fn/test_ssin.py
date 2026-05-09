"""Tests for moirais.fn.ssin — Sample size for non-inferiority."""

import pytest

from moirais.fn.ssin import sample_size_noninferiority


class TestSampleSizeNoninferiority:
    def test_basic(self):
        res = sample_size_noninferiority(0.0, 0.5, 1.0)
        assert res.estimate > 10

    def test_positive_delta(self):
        r1 = sample_size_noninferiority(0.0, 0.5, 1.0)
        r2 = sample_size_noninferiority(0.2, 0.5, 1.0)
        assert r2.estimate <= r1.estimate

    def test_invalid_margin(self):
        with pytest.raises(ValueError):
            sample_size_noninferiority(0.0, -0.5, 1.0)
