"""Web page fetch + text extraction. 'Patience is bitter, but its fruit is sweet. -- Aristotle'"""
from __future__ import annotations

import urllib.request
from html.parser import HTMLParser

from ._containers import DescriptiveResult


class _TextExtractor(HTMLParser):
    """Strip HTML tags and collect text content."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def web_scrape(url: str, timeout: int = 30) -> str:
    """Fetch a web page and return raw HTML.

    Parameters
    ----------
    url : str
        URL to fetch.
    timeout : int, default 30
        Request timeout in seconds.

    Returns
    -------
    str
        Raw HTML content.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "morie/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_text(html: str) -> str:
    """Strip HTML tags and return plain text.

    Parameters
    ----------
    html : str

    Returns
    -------
    str
    """
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text().strip()


def web_scrape_result(url: str, timeout: int = 30) -> DescriptiveResult:
    """Fetch URL and return both HTML and extracted text."""
    html = web_scrape(url, timeout=timeout)
    text = extract_text(html)
    return DescriptiveResult(
        name="Web scrape",
        value=text,
        extra={"url": url, "html_length": len(html), "text_length": len(text)},
    )


wscrp = web_scrape_result


def cheatsheet() -> str:
    return "web_scrape({}) -> Web page fetch + text extraction. 'Punch it, Chewie!' -- Han"
