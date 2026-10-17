"""Scrape FOMC statements from federalreserve.gov.

Where statements live has changed over the years:

* the current calendar page (``fomccalendars.htm``) covers the last five or so
  years and links each statement as "HTML" ->
  ``/newsevents/pressreleases/monetaryYYYYMMDDa.htm``;
* older years each have a page ``fomchistoricalYYYY.htm`` with links titled
  "Statement", pointing at ``/newsevents/press/monetary/YYYYMMDDa.htm``
  (2006-2008, now redirected), ``/boarddocs/press/monetary/YYYY/YYYYMMDD/``
  (2002-2005) or ``/boarddocs/press/general/YYYY/YYYYMMDD/`` (earlier).

Statement pages also come in two layouts: the modern one keeps the text in
``#article > div.col-sm-8`` (the first such div is the heading), while the
boarddocs pages are bare paragraphs.
"""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
CALENDAR = BASE + "/monetarypolicy/fomccalendars.htm"
HISTORICAL = BASE + "/monetarypolicy/fomchistorical{year}.htm"

_MODERN = re.compile(r"/newsevents/press(?:releases)?/monetary/?(\d{8})([a-z])\.htm$")
_BOARDDOCS = re.compile(r"/boarddocs/press/(?:monetary|general)/\d{4}/(\d{8})/?(?:default\.htm)?$")

_BOILERPLATE = re.compile(
    r"^(for immediate release|for release at|for media inquiries|implementation note issued|"
    r"release date|last update|return to top|\d{4} monetary policy$|home \|)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StatementLink:
    date: date
    url: str


def normalise_url(href: str) -> str | None:
    """Canonical absolute URL for a statement link, or None if it isn't one."""
    path = href.replace(BASE, "")
    m = _MODERN.search(path)
    if m:
        return f"{BASE}/newsevents/pressreleases/monetary{m.group(1)}{m.group(2)}.htm"
    m = _BOARDDOCS.search(path)
    if m:
        d = m.group(1)
        kind = "monetary" if "/monetary/" in path else "general"
        return f"{BASE}/boarddocs/press/{kind}/{d[:4]}/{d}/"
    return None


def date_from_url(url: str) -> date:
    d = re.search(r"(\d{8})", url).group(1)
    return date(int(d[:4]), int(d[4:6]), int(d[6:]))


def statement_links(html: str) -> list[StatementLink]:
    """Statement links on a calendar or historical page."""
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, StatementLink] = {}
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        href = a["href"]
        url = normalise_url(href)
        if url is None:
            continue
        is_modern_statement = bool(re.search(r"monetary\d{8}a\.htm$", href)) and text == "html"
        if text == "statement" or is_modern_statement:
            found[url] = StatementLink(date_from_url(url), url)
    return sorted(found.values(), key=lambda s: s.date)


def calendar_years(html: str) -> list[int]:
    """Years shown on the current calendar page (from its statement links)."""
    return sorted({s.date.year for s in statement_links(html)})


def _paragraph_texts(node) -> list[str]:
    """Text of each <p>, without double counting nested paragraphs.

    Old pages sometimes leave <p> tags unclosed, so Python's parser nests the
    following paragraphs inside them. For such a <p> only its own direct text
    is kept; the nested paragraphs are picked up on their own.
    """
    out = []
    for p in node.find_all("p"):
        if p.find("p") is None:
            out.append(p.get_text(" ", strip=True))
        else:
            out.append(" ".join(s.strip() for s in p.find_all(string=True, recursive=False)))
    return out


def extract_text(html: str) -> str:
    """Statement body as plain paragraphs separated by blank lines."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("#article")
    if article is not None:
        blocks = [d for d in article.select("div.col-sm-8") if "heading" not in (d.get("class") or [])]
        paras = [t for d in blocks for t in _paragraph_texts(d)]
    else:
        paras = _paragraph_texts(soup)
    out, seen = [], set()
    for p in paras:
        p = re.sub(r"\s+", " ", p).strip()
        if not p or _BOILERPLATE.search(p) or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return "\n\n".join(out)


def _get(url: str, timeout: float = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _cached_get(url: str, cache_dir: Path, pause: float) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", url.replace(BASE, "")).strip("_") + ".html"
    path = cache_dir / name
    if path.exists():
        return path.read_text()
    html = _get(url)
    path.write_text(html)
    time.sleep(pause)
    return html


def fetch_statements(start_year: int = 2000, cache_dir: str | Path = "data/raw", pause: float = 0.3) -> pd.DataFrame:
    """Download every FOMC statement from ``start_year`` on.

    Raw HTML is cached, so re-running only fetches new statements.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cal_html = _get(CALENDAR)
    links = statement_links(cal_html)
    first_cal_year = min(calendar_years(cal_html))
    for year in range(start_year, first_cal_year):
        links += statement_links(_cached_get(HISTORICAL.format(year=year), cache, pause))

    rows = {}
    for link in links:
        if link.date.year < start_year:
            continue
        text = extract_text(_cached_get(link.url, cache, pause))
        if len(text) < 200:  # skip pages that aren't really statements
            continue
        rows[link.date] = {"date": pd.Timestamp(link.date), "url": link.url, "text": text}
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["date"]))
    df["n_words"] = df["text"].str.split().str.len()
    return df
