"""Tests for moirais.siu offline pieces (no network).

Phase 2a covers the parser, schema, normalizer, and writer. Phase 2b
will add live-fetch tests against 5 sample drids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moirais.siu import (
    BLANK_ROW,
    SIU_COLUMNS,
    parse_html,
    write_csv,
    write_jsonl,
)
from moirais.siu._normalize import (
    find_case_number,
    parse_date,
    parse_drid_from_url,
    parse_nrid_from_url,
    normalise_sex,
    normalise_yes_no,
)


# Synthetic minimal HTML matching the SIU page conventions.
SAMPLE_HTML = """
<!DOCTYPE html>
<html><body>
<main>
<h1>SIU Director's Report</h1>
<p>Case Number: 17-PVI-371</p>
<p>Date of Incident: January 5, 2017</p>
<p>Date SIU was Notified: January 6, 2017</p>
<p>Date of SIU Director's Decision: April 12, 2017</p>
<p>Police Service: Toronto Police Service</p>
<p>Number of Officers: 2</p>
<p>Location: 100 Queen Street West, Toronto</p>
<p>Reason for Interaction: Vehicle stop</p>
<p>Sex: Male</p>
<p>Age: 34</p>
<p>Injuries Sustained: Fractured rib</p>
<p>Charges Recommended: No</p>
<p>The Investigation</p>
<p>The investigation determined that the individual, who had a history of
mental health issues including schizophrenia, was approached by officers
following a vehicle stop. Body-worn camera footage was reviewed.</p>
<p>Director's Decision</p>
<p>Reasonable grounds were not met to lay any criminal charges.</p>
<p><a href="news_template.php?nrid=4494">News release</a></p>
<p><a href="https://news.example.com/story-123">Coverage</a></p>
</main>
</body></html>
"""


def test_schema_has_45_cols():
    # Count is 60 actually because we keep iso/raw twice for each temporal
    # field — the "45-col target" was approximate. Verify >= 45 and that
    # all canonical keys present.
    assert "case_number" in SIU_COLUMNS
    assert "drid" in SIU_COLUMNS
    assert "nrid" in SIU_COLUMNS
    assert "narrative_full" in SIU_COLUMNS
    assert "scraped_at_utc" in SIU_COLUMNS
    assert len(SIU_COLUMNS) >= 45
    # BLANK_ROW must cover all columns
    assert set(BLANK_ROW.keys()) == set(SIU_COLUMNS)


def test_parse_html_extracts_case_number():
    row = parse_html(SAMPLE_HTML, drid=80,
                     source_url="https://www.siu.on.ca/en/directors_report_details.php?drid=80")
    assert row["case_number"] == "17-PVI-371"
    assert row["drid"] == 80
    assert row["nrid"] == 4494
    assert row["police_service"] == "Toronto Police Service"
    assert row["number_of_officers_involved"] == 2
    assert row["sex_gender_affected"] == "male"
    assert row["age_affected"] == "34"
    assert row["charges_recommended"] is False


def test_parse_html_dates_normalised():
    row = parse_html(SAMPLE_HTML)
    assert row["date_of_incident_iso"] == "2017-01-05"
    assert row["date_of_incident_raw"] == "January 5, 2017"
    assert row["date_siu_notified_iso"] == "2017-01-06"
    assert row["date_of_director_decision_iso"] == "2017-04-12"


def test_parse_html_narrative_preserved():
    row = parse_html(SAMPLE_HTML)
    assert row["narrative_full"]
    assert "schizophrenia" in row["narrative_full"]
    assert "Reasonable grounds" in row["narrative_full"]


def test_parse_html_mh_race_indicators():
    row = parse_html(SAMPLE_HTML)
    sig = row["mental_health_or_race_indications"]
    assert "mental health" in sig
    assert "schizophren" in sig


def test_parse_html_supplemental_links():
    row = parse_html(SAMPLE_HTML, source_url="https://www.siu.on.ca/en/directors_report_details.php?drid=80")
    # External links go to supplemental_materials, not internal SIU links.
    assert "news.example.com/story-123" in row["supplemental_materials"]
    # The internal nrid link goes into source_url_news, not supplemental.
    assert row["source_url_news"]
    assert "nrid=4494" in row["source_url_news"]
    assert "siu.on.ca" not in row["supplemental_materials"]


def test_parse_html_missing_fields_become_none():
    row = parse_html("<html><body>nothing here</body></html>", drid=999)
    assert row["case_number"] is None
    assert row["police_service"] is None
    assert row["date_of_incident_iso"] is None
    assert row["drid"] == 999
    assert row["parser_version"]


# ── normalize unit tests ──────────────────────────────────────────────────

def test_find_case_number_label_form():
    assert find_case_number("Case Number: 17-PVI-371") == "17-PVI-371"
    assert find_case_number("Case # 21-TCI-045") == "21-TCI-045"
    assert find_case_number("SIU Case Number 24-OCI-123") == "24-OCI-123"


def test_find_case_number_bare_form():
    assert find_case_number("blah blah 17-PVI-371 elsewhere") == "17-PVI-371"


def test_find_case_number_uppercase_normalises():
    # SIU codes are uppercase; we should normalise even if input is lowercase.
    assert find_case_number("Case# 17-pvi-371") == "17-PVI-371"


def test_find_case_number_returns_none_for_no_match():
    assert find_case_number("nothing in this text") is None


def test_parse_date_long_form():
    iso, raw = parse_date("January 5, 2017")
    assert iso == "2017-01-05"
    assert raw == "January 5, 2017"


def test_parse_date_iso_form():
    iso, raw = parse_date("2017-01-05")
    assert iso == "2017-01-05"


def test_parse_date_unparseable_keeps_raw():
    iso, raw = parse_date("sometime last winter")
    assert iso is None
    assert raw == "sometime last winter"


def test_parse_drid_nrid_from_url():
    assert parse_drid_from_url("https://x/?drid=80") == 80
    assert parse_nrid_from_url("https://x/?nrid=4494") == 4494
    assert parse_drid_from_url("no-drid-here") is None


def test_normalise_sex_synonyms():
    assert normalise_sex("Male") == "male"
    assert normalise_sex("F") == "female"
    assert normalise_sex("non-binary") == "nonbinary"
    assert normalise_sex("Unknown") == "unknown"
    assert normalise_sex("") is None


def test_normalise_yes_no():
    assert normalise_yes_no("Yes") is True
    assert normalise_yes_no("Charged") is True
    assert normalise_yes_no("No charges") is False
    assert normalise_yes_no("False") is False
    assert normalise_yes_no("maybe") is None


# ── writer tests ──────────────────────────────────────────────────────────

def test_write_csv_round_trip(tmp_path: Path):
    rows = [parse_html(SAMPLE_HTML, drid=80,
                        source_url="https://www.siu.on.ca/en/directors_report_details.php?drid=80")]
    out = tmp_path / "siu.csv"
    n = write_csv(rows, out)
    assert n == 1
    assert out.exists()
    content = out.read_text()
    assert "case_number" in content
    assert "17-PVI-371" in content
    # narrative_full is excluded by default — narrative_summary is NOT,
    # which is fine (it's intended to be the short queryable form).
    # We assert the CSV does NOT have the literal "narrative_full" header.
    # Header line is the first line.
    header = content.splitlines()[0]
    assert "narrative_summary" in header
    assert "narrative_full" not in header


def test_write_csv_with_narrative(tmp_path: Path):
    rows = [parse_html(SAMPLE_HTML)]
    out = tmp_path / "siu_full.csv"
    write_csv(rows, out, exclude_narrative=False)
    assert "schizophrenia" in out.read_text()


def test_write_jsonl(tmp_path: Path):
    rows = [parse_html(SAMPLE_HTML, drid=80)]
    out = tmp_path / "siu_narratives.jsonl"
    n = write_jsonl(rows, out)
    assert n == 1
    obj = json.loads(out.read_text().splitlines()[0])
    assert obj["case_number"] == "17-PVI-371"
    assert obj["drid"] == 80
    assert "schizophrenia" in obj["narrative_full"]


# ── Real-fixture tests: drid=80 (rendered text from siu.on.ca) ─────────
# https://www.siu.on.ca/en/directors_report_details.php?drid=80 on
# 2026-05-06. The fixture lives next to this test file; loading it
# into a <main>-wrapped HTML reproduces the parser's actual workload.

FIXTURES = Path(__file__).parent / "fixtures" / "siu"


def _load_real_drid(drid: int) -> dict:
    fixture = (FIXTURES / f"drid_{drid}_rendered.txt").read_text()
    html = f"<html><body><main>{fixture}</main></body></html>"
    return parse_html(
        html, drid=drid,
        source_url=f"https://www.siu.on.ca/en/directors_report_details.php?drid={drid}",
    )


def test_real_drid_80_canonical_fields():
    """drid=80 ↔ Case 17-PVI-371 ↔ nrid=4494 (Bonfield collision, OPP).

    All 19 expected fields must match the page's verbatim content. This
    is the parser's gold-standard regression — break this and you've
    broken something for every report drid.
    """
    row = _load_real_drid(80)

    assert row["case_number"] == "17-PVI-371"
    assert row["drid"] == 80
    assert row["police_service"] == "Ontario Provincial Police"
    assert row["notifying_party"] == "Ontario Provincial Police"

    # Temporal — all three dates parse to ISO
    assert row["date_of_incident_iso"] == "2017-12-22"
    assert row["date_siu_notified_iso"] == "2017-12-22"
    assert row["date_of_director_decision_iso"] == "2018-10-18"

    # Location + counts
    assert row["location_of_call"] == "Township of Bonfield"
    assert row["number_of_officers_involved"] == 1
    assert row["number_of_civilian_witnesses"] == 3
    assert row["number_of_witness_officials"] == 0
    assert row["number_of_subject_officials"] == 1
    assert row["subject_official_interviewed_or_notes"] == "Yes"
    assert row["siu_investigators"] == "2"
    assert row["siu_forensics_investigators"] == "1"

    # Affected person
    assert row["sex_gender_affected"] == "female"
    assert row["age_affected"] == "25"

    # Legislation cited
    assert "Criminal Code" in (row["relevant_legislation"] or "")
    assert "Highway Traffic Act" in (row["relevant_legislation"] or "")

    # Decision: no charges
    assert row["charges_recommended"] is False


def test_real_drid_80_narrative_excludes_navigation_chrome():
    """The TOC at top of every SIU page lists "First Nations, Inuit and
    Métis Liaison Program" in the menu. That MUST NOT leak into
    `narrative_full` (or it would falsely trigger the MH/race scanner)."""
    row = _load_real_drid(80)
    nf = row["narrative_full"] or ""
    # body section starts with Mandate engaged or similar
    assert nf.lstrip().startswith(("Mandate engaged", "The Investigation",
                                    "Notification of the SIU"))
    # MH/race indicator should NOT include nav-menu phrases for drid=80
    # which has no actual MH/race content in its narrative.
    assert "First Nations" not in (row["mental_health_or_race_indications"] or "")
    assert "Métis" not in (row["mental_health_or_race_indications"] or "")


def test_real_drid_80_does_not_leak_toc_into_section_extraction():
    """The 'Contents:' table-of-contents at the top of every page lists
    section names. `_section_text` must skip the TOC and slice into the
    actual body section. drid=80's Incident Narrative starts with "On
    December 22, 2017" — if our slicer picks the TOC entry, the date
    won't be found."""
    row = _load_real_drid(80)
    assert row["date_of_incident_iso"] == "2017-12-22", \
        "incident date came from Incident Narrative section, not TOC"


# ── Real-fixture tests: drid=5074 (post-2019 modern format) ────────────
# This page is the most-recent SIU report at the time of writing and
# uses the post-Special-Investigations-Unit-Act-2019 format with
# renamed sections: "Subject Official" (was "Subject Officers"),
# "Witness Officials" (was "Witness Officers").

def test_real_drid_5074_modern_format_canonical_fields():
    """drid=5074 ↔ Case 26-OCI-016 ↔ NRPS, Director Joseph Martino.

    Tests every field that should round-trip on the modern post-2019
    format. Breaking this means we've regressed on every report
    written after the SIU Act 2019 took effect.
    """
    row = _load_real_drid(5074)

    assert row["case_number"] == "26-OCI-016"
    assert row["drid"] == 5074
    assert row["police_service"] == "Niagara Regional Police Service"
    assert row["notifying_party"] == "Niagara Regional Police Service"

    # Temporal — all three dates parse to ISO
    assert row["date_of_incident_iso"] == "2026-01-11"
    assert row["date_siu_notified_iso"] == "2026-01-12"
    assert row["date_of_director_decision_iso"] == "2026-05-05"

    # Counts — modern format adds Witness Officials (was Officers)
    assert row["number_of_officers_involved"] == 1
    assert row["number_of_witness_officials"] == 4
    assert row["number_of_subject_officials"] == 1
    assert row["subject_official_interviewed_or_notes"] == "Yes"
    assert row["siu_investigators"] == "3"
    assert row["siu_forensics_investigators"] == "0"

    # Affected person
    assert row["sex_gender_affected"] == "male"
    assert row["age_affected"] == "34"

    # Decision — no charges; modern format
    assert row["charges_recommended"] is False
    assert row["_language"] == "en"


def test_incident_date_skips_notification_sentence():
    """Modern reports open the Investigation with two dates in adjacent
    sentences: the SIU contacted/notified date, then the actual incident
    date. The parser must NOT pick the first one as date_of_incident.
    drid=5074: SIU contacted Jan 12, incident was Jan 11 → expect 11.
    """
    row = _load_real_drid(5074)
    assert row["date_of_incident_iso"] == "2026-01-11"
    assert row["date_siu_notified_iso"] == "2026-01-12"


# ── News-release parser tests (real nrid fixtures) ─────────────────────

from moirais.siu import parse_news_html


def _load_real_nrid(nrid: int) -> dict:
    fixture = (FIXTURES / f"nrid_{nrid}_rendered.txt").read_text()
    html = f"<html><body><main>{fixture}</main></body></html>"
    return parse_news_html(
        html, nrid=nrid,
        source_url=f"https://www.siu.on.ca/en/news_template.php?nrid={nrid}",
    )


def test_real_nrid_4494_pre_2019_news_release():
    """nrid=4494 ↔ Case 17-PVI-371 ↔ drid=80 (Director Tony Loparco)."""
    out = _load_real_nrid(4494)
    assert out["case_number"] == "17-PVI-371"
    assert out["news_release_title"].startswith("No Basis to Charge OPP Officer")
    assert out["news_release_date_iso"] == "2018-12-06"
    assert out["news_release_date_raw"] == "6 December, 2018"
    assert out["directors_name"] == "Tony Loparco"
    assert "On December 22, 2017" in (out["news_release_summary"] or "")


def test_real_nrid_11177_2026_news_release():
    """nrid=11177 ↔ Case 26-OCI-016 ↔ drid=5074 (Director Joseph Martino).

    Tests that the news template format is stable across the 2018→2026
    span: same headline-then-date pattern, same director-name pattern.
    """
    out = _load_real_nrid(11177)
    assert out["case_number"] == "26-OCI-016"
    assert "Broken Orbital Bone" in (out["news_release_title"] or "")
    assert out["news_release_date_iso"] == "2026-05-05"
    assert out["news_release_date_raw"] == "5 May, 2026"
    assert out["directors_name"] == "Joseph Martino"
