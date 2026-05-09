"""Tests for moirais.fn.trnty -- CVSS v3.1 base score."""

from moirais.fn.trnty import cvss_base, trnty
from moirais.fn._containers import DescriptiveResult


class TestTrnty:
    def test_alias(self):
        assert trnty is cvss_base

    def test_critical_score(self):
        result = cvss_base(av="N", ac="L", pr="N", ui="N", s="C", c="H", i="H", a="H")
        assert isinstance(result, DescriptiveResult)
        assert result.value == 10.0

    def test_low_score(self):
        result = cvss_base(av="P", ac="H", pr="H", ui="R", s="U", c="N", i="N", a="N")
        assert result.value == 0.0

    def test_vector_string(self):
        result = cvss_base(av="N", ac="L", pr="L", ui="N", s="U", c="L", i="L", a="N")
        assert "CVSS:3.1" in result.extra["vector"]
