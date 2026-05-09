"""Canonical 45-column schema for SIU.csv.

Built from the union of SIU1a (Qualtrics 1463-row export, 30 substantive
cols) and SIU1b/Sheet1 (the improvement attempt with 57 cols including
time-of-incident, type-of-building, specific-injuries).

Every temporal field is stored TWICE:
    <field>_iso  = normalized ISO-8601
    <field>_raw  = exact verbatim string from the report page

This addresses the author's SIU1a complaint that "variables weren't properly
input, like the numerical formatting for temporal variables", without
ever losing audit-trail data.
"""

from __future__ import annotations

# Column order matters — this is the CSV header order.
SIU_COLUMNS: list[str] = [
    # ── Identifiers / provenance ───────────────────────────────────
    "case_number",                  # PRIMARY KEY  e.g. "17-PVI-371"
    "drid",                         # URL locator (integer)
    "nrid",                         # news-release URL locator (integer or null)
    "source_url_report",
    "source_url_news",
    "scraped_at_utc",               # ISO-8601 with 'Z' suffix
    "parser_version",               # semver of the parser at extract time

    # ── Temporal — incident ─────────────────────────────────────────
    "date_of_incident_iso",
    "date_of_incident_raw",
    "time_of_incident_raw",
    "date_of_injury_iso",
    "date_of_injury_raw",
    "incident_to_injury_raw",       # duration string

    # ── Temporal — notification + decision ──────────────────────────
    "date_siu_notified_iso",
    "date_siu_notified_raw",
    "time_of_notification_raw",
    "notifying_party",
    "notifying_party_other_text",
    "date_of_director_decision_iso",
    "date_of_director_decision_raw",
    "time_of_director_decision_raw",
    "siu_investigators",            # free-text or "; "-joined names
    "siu_forensics_investigators",

    # ── Police / scene ──────────────────────────────────────────────
    "police_service",
    "number_of_officers_involved",
    "location_of_call",
    "type_of_building_or_scene",
    "reason_for_interaction",

    # ── Affected person ─────────────────────────────────────────────
    "injuries_sustained",
    "injuries_other_text",
    "specific_injuries",
    "location_of_treatment",
    "number_of_affected_persons",
    "sex_gender_affected",
    "age_affected",
    "affected_interviewed",
    "date_of_affected_interview_iso",
    "date_of_affected_interview_raw",

    # ── Witnesses + officer interviews ──────────────────────────────
    "number_of_civilian_witnesses",
    "date_of_witness_interview_raw",
    "number_of_subject_officials",
    "subject_official_interviewed_or_notes",
    "date_of_subject_interview_raw",
    "number_of_witness_officials",
    "date_of_witness_official_interview_raw",

    # ── Evidence + decision ─────────────────────────────────────────
    "evidence_types",                # joined "; "
    "evidence_other_text",
    "evidence_features",
    "narrative_summary",             # ~1 paragraph
    "narrative_full",                # full director's report body
    "relevant_legislation",
    "legislation_other_text",
    "weapons_or_force_used",
    "weapons_other_text",
    "charges_recommended",           # bool-like
    "directors_decision_reasonable",

    # ── Supplemental ────────────────────────────────────────────────
    "supplemental_materials",
    "news_links_extra",
    "mental_health_or_race_indications",

    # ── Provenance flags (added Phase 2b, 2026-05-06) ───────────────
    "_language",                     # 'en' | 'fr' | 'unknown'

    # ── News release (parsed from paired nrid page, 2026-05-06) ─────
    "news_release_title",            # headline of the release
    "news_release_date_iso",         # release publication date
    "news_release_date_raw",
    "news_release_summary",          # 1-paragraph public summary
    "directors_name",                # director who signed the report
]


# A row dict pre-populated with None for every column. parse_html() returns
# this shape; missing fields stay None rather than KeyError.
BLANK_ROW: dict[str, object] = {col: None for col in SIU_COLUMNS}
