"""HTTP fetch layer for SIU mining -- caching + politeness + retries.

Sequential by default (single in-flight request, polite 250 ms gap)
because Phase 2a is a scaffold. Phase 2c will turn on async concurrency
behind the same public surface (`scrape_drid`, `scrape_range`).

Caching policy:
    cache hits: served from `data/cache/siu/<drid>.html`
    404 sentinels: zero-byte file at `<drid>.404` so we never refetch
    forced refresh: pass cache=False
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import httpx
import stamina

from ._parser import parse_html, parse_news_html, PARSER_VERSION
from ._schema import BLANK_ROW

SIU_BASE = "https://www.siu.on.ca"
REPORT_URL = SIU_BASE + "/en/directors_report_details.php?drid={drid}"
NEWS_URL = SIU_BASE + "/en/news_template.php?drid={drid}"

# Default cache lives at the project's `data/cache/siu/` -- same root the
# rest of morie uses for pickled / RDS / SQLite caches.
# __file__ is .../libexec/config/tools/py-package/morie/siu/_scraper.py
# parents[6] climbs back to dev/sphinx/project/.
DEFAULT_CACHE = Path(__file__).resolve().parents[6] / "data" / "cache" / "siu"

USER_AGENT = (
    "morie-siu-scraper/0.1 "
    "(+https://github.com/hadesllm/morie; research; rate-limited)"
)
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
POLITE_DELAY_S = 0.25       # sequential floor between requests
RETRY_ATTEMPTS = 3


def _cache_path(cache_dir: Path, drid: int, suffix: str = ".html") -> Path:
    return cache_dir / f"{drid}{suffix}"


@stamina.retry(
    on=(httpx.TransportError, httpx.HTTPStatusError),
    attempts=RETRY_ATTEMPTS,
    wait_initial=1.0,
    wait_max=15.0,
    wait_jitter=1.5,
)
def _fetch(client: httpx.Client, url: str) -> httpx.Response:
    """Single GET with retry on transport / 5xx errors. 404 is NOT retried."""
    r = client.get(url, follow_redirects=True)
    if r.status_code == 404:
        return r
    r.raise_for_status()
    return r


def scrape_drid(
    drid: int,
    *,
    client: Optional[httpx.Client] = None,
    cache: bool = True,
    cache_dir: Optional[Path] = None,
    fetch_news: bool = True,
) -> dict:
    """Fetch + parse one director's report. Returns a SIU.csv row dict.

    On 404, returns a row with case_number=None and the source_url set --
    the caller decides whether to drop it.

    If `fetch_news=True` (default) and the report links to a news release
    via `news_template.php?nrid=…`, that page is also fetched and its
    fields are merged into the row (news_release_title, _date_iso, _raw,
    _summary, directors_name).
    """
    cache_dir = (cache_dir or DEFAULT_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)

    html_path = _cache_path(cache_dir, drid, ".html")
    sentinel = _cache_path(cache_dir, drid, ".404")
    url = REPORT_URL.format(drid=drid)

    if cache and sentinel.exists():
        return _empty_404_row(drid, url)

    owns = client is None
    if owns and not (cache and html_path.exists()):
        client = httpx.Client(timeout=DEFAULT_TIMEOUT,
                              headers={"User-Agent": USER_AGENT})

    try:
        if cache and html_path.exists():
            html = html_path.read_text(encoding="utf-8", errors="replace")
        else:
            r = _fetch(client, url)
            if r.status_code == 404:
                sentinel.write_bytes(b"")
                return _empty_404_row(drid, url)
            html = r.text
            tmp = html_path.with_suffix(".html.tmp")
            tmp.write_text(html, encoding="utf-8")
            tmp.replace(html_path)

        row = parse_html(html, drid=drid, source_url=url)

        # ── paired news release fetch + merge ───────────────────────
        if fetch_news and row.get("nrid"):
            try:
                news = _scrape_news(row["nrid"], client=client,
                                    cache=cache, cache_dir=cache_dir)
                # Cross-validate case_number; warn if mismatch
                if news.get("case_number") and row.get("case_number") and \
                   news["case_number"] != row["case_number"]:
                    row["_news_case_number_mismatch"] = news["case_number"]
                # Merge news-page fields (don't overwrite if already set)
                for k in ("news_release_title", "news_release_date_iso",
                          "news_release_date_raw", "news_release_summary",
                          "directors_name"):
                    if news.get(k) and not row.get(k):
                        row[k] = news[k]
            except Exception as e:  # noqa: BLE001
                row["_news_fetch_error"] = f"{type(e).__name__}: {e}"

        row["scraped_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return row
    finally:
        if owns and client is not None:
            client.close()


def _scrape_news(nrid: int,
                 *, client: Optional[httpx.Client] = None,
                 cache: bool = True,
                 cache_dir: Optional[Path] = None) -> dict:
    """Fetch + parse one news-release page (`news_template.php?nrid=N`)."""
    cache_dir = (cache_dir or DEFAULT_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    html_path = cache_dir / f"news_{nrid}.html"
    url = f"{SIU_BASE}/en/news_template.php?nrid={nrid}"

    if cache and html_path.exists():
        html = html_path.read_text(encoding="utf-8", errors="replace")
    else:
        owns = client is None
        if owns:
            client = httpx.Client(timeout=DEFAULT_TIMEOUT,
                                  headers={"User-Agent": USER_AGENT})
        try:
            r = _fetch(client, url)
            if r.status_code == 404:
                return {"nrid": nrid, "source_url_news": url}
            html = r.text
            tmp = html_path.with_suffix(".html.tmp")
            tmp.write_text(html, encoding="utf-8")
            tmp.replace(html_path)
        finally:
            if owns:
                client.close()

    return parse_news_html(html, nrid=nrid, source_url=url)


def scrape_range(
    drid_min: int,
    drid_max: int,
    *,
    cache: bool = True,
    cache_dir: Optional[Path] = None,
    delay_s: float = POLITE_DELAY_S,
    progress: bool = True,
) -> Iterator[dict]:
    """Yield rows for every drid in [drid_min, drid_max] inclusive.

    Sequential, polite (sleeps `delay_s` between fetches that touch the
    network -- cache hits don't sleep). Phase-2a default; Phase-2c will
    introduce a concurrent variant behind the same name.
    """
    with httpx.Client(timeout=DEFAULT_TIMEOUT,
                      headers={"User-Agent": USER_AGENT}) as client:
        last_was_network = False
        for drid in range(drid_min, drid_max + 1):
            if last_was_network:
                time.sleep(delay_s)
            html_path = _cache_path((cache_dir or DEFAULT_CACHE), drid, ".html")
            sentinel = _cache_path((cache_dir or DEFAULT_CACHE), drid, ".404")
            cache_hit = cache and (html_path.exists() or sentinel.exists())
            row = scrape_drid(drid, client=client, cache=cache, cache_dir=cache_dir)
            last_was_network = not cache_hit
            if progress:
                cn = row.get("case_number") or "--"
                print(f"  drid={drid:5}  case={cn}  cached={cache_hit}")
            yield row


def _empty_404_row(drid: int, url: str) -> dict:
    row = dict(BLANK_ROW)
    row["drid"] = drid
    row["source_url_report"] = url
    row["parser_version"] = PARSER_VERSION
    row["scraped_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return row


def check_robots_txt(*, client: Optional[httpx.Client] = None) -> dict:
    """Fetch siu.on.ca/robots.txt and return parsed disallow rules.

    Call this once at the start of any large scrape. If the site disallows
    the report path, abort the scrape rather than crawl anyway.
    """
    owns = client is None
    if owns:
        client = httpx.Client(timeout=DEFAULT_TIMEOUT,
                              headers={"User-Agent": USER_AGENT})
    try:
        r = client.get(SIU_BASE + "/robots.txt", follow_redirects=True)
        if r.status_code != 200:
            return {"status": r.status_code, "text": "", "disallows": []}
        text = r.text
        disallows = []
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("disallow:"):
                disallows.append(line.split(":", 1)[1].strip())
        return {"status": 200, "text": text, "disallows": disallows}
    finally:
        if owns:
            client.close()
