from datetime import date
from pathlib import Path

from fomc_signal.fed import BASE, extract_text, normalise_url, statement_links

FIX = Path(__file__).parent / "fixtures"


def test_calendar_page_links_only_statements():
    links = statement_links((FIX / "calendar.html").read_text())
    assert [l.date for l in links] == [date(2025, 12, 10), date(2026, 1, 28)]
    assert links[1].url == BASE + "/newsevents/pressreleases/monetary20260128a.htm"


def test_historical_page_all_url_generations():
    links = statement_links((FIX / "historical.html").read_text())
    urls = [l.url for l in links]
    assert urls == [
        BASE + "/boarddocs/press/general/2000/20000202/",
        BASE + "/boarddocs/press/monetary/2003/20030129/",
        BASE + "/newsevents/pressreleases/monetary20080122b.htm",
        BASE + "/newsevents/pressreleases/monetary20080130a.htm",
    ]


def test_normalise_rejects_other_pages():
    assert normalise_url("/monetarypolicy/fomcminutes20080130.htm") is None
    assert normalise_url("/newsevents/pressreleases/monetary20260128a1.htm") is None


def test_extract_modern_layout_drops_heading_and_boilerplate():
    text = extract_text((FIX / "statement_modern.html").read_text())
    paras = text.split("\n\n")
    assert paras[0].startswith("Available indicators")
    assert len(paras) == 3
    assert "media" not in text and "Implementation Note" not in text and "January 28" not in text


def test_extract_boarddocs_dedupes_nested_paragraphs():
    text = extract_text((FIX / "statement_boarddocs.html").read_text())
    paras = text.split("\n\n")
    assert len(paras) == 2
    assert paras[0].startswith("The Federal Open Market Committee voted today to raise")
    assert "Monetary policy" not in text
