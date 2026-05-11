"""Tests for morie.fn.siubc -- SIU Ontario report parser."""

from morie.fn.siubc import siu_scrape_report, siubc
from morie.fn._containers import DescriptiveResult


_SAMPLE_HTML = """
<html><body>
<h1>SIU Investigation</h1>
<p>Case No. 22-TCI-123</p>
<p>Date: January 15, 2023</p>
<p>Subject Officer: Constable J Smith</p>
<p>Allegations: Excessive use of force during arrest</p>
<p>Findings: No reasonable grounds to charge</p>
</body></html>
"""


class TestSiubc:
    def test_alias(self):
        assert siubc is siu_scrape_report

    def test_parse_fields(self):
        result = siu_scrape_report(_SAMPLE_HTML)
        assert isinstance(result, DescriptiveResult)
        v = result.value
        assert v["case_number"] == "22-TCI-123"
        assert "2023" in v["date"]

    def test_missing_fields(self):
        result = siu_scrape_report("<html><body>Nothing here</body></html>")
        assert result.value["case_number"] is None
