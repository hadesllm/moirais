"""Tests for morie.fn.wscrp -- Web scrape + text extraction."""

from io import BytesIO
from unittest.mock import patch, MagicMock

from morie.fn.wscrp import web_scrape, extract_text, wscrp


class TestWscrp:
    def test_alias(self):
        assert wscrp is not web_scrape

    def test_extract_text(self):
        html = "<html><body><p>Hello</p><script>var x=1;</script><p>World</p></body></html>"
        text = extract_text(html)
        assert "Hello" in text
        assert "World" in text
        assert "var x" not in text

    def test_web_scrape_mocked(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><body>OK</body></html>"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("morie.fn.wscrp.urllib.request.urlopen", return_value=mock_resp):
            html = web_scrape("http://example.com")
            assert "OK" in html
