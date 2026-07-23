"""Regex-based extraction of defined terms / abbreviations from narrative text.

Two sources are combined: (1) inline definitions embedded in prose anywhere in the
document, and (2) a dedicated Glossary/Definitions/Abbreviations section, if one
exists, parsed as a flat list of term/definition pairs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .narrative import ParagraphUnit
from .schema import GlossaryEntry

GLOSSARY_SECTION_RE = re.compile(r"glossary|definitions|abbreviations|defined terms", re.IGNORECASE)

# "X" means / refers to / is defined as ...
QUOTED_DEFINES_RE = re.compile(
    r'["“]([A-Z][\w &/,.\-]{1,60})["”]\s+'
    r"(?:means|refers to|is defined as|shall mean|includes)\s+([^.]+)\.",
)

# Long form name ("ABBR") — the common SEC-filing pattern introducing an abbreviation.
PAREN_ABBR_RE = re.compile(r'([A-Z][A-Za-z0-9&,.\- ]{3,80}?)\s*\(["“]?([A-Z][A-Z0-9&]{1,9})["”]?\)')

# "ABBR" means/refers to ... (abbreviation defined by its own definition, not parenthetical)
ABBR_MEANS_RE = re.compile(
    r'["“]([A-Z]{2,10})["”]\s+(?:as used (?:herein|in this report),?\s+)?'
    r"(?:means|refers to)\s+([^.]+)\.",
)

# Best-effort flat "Term — definition. Term2 — definition2." splitter for dedicated
# glossary sections, where line breaks have already been collapsed into one paragraph.
GLOSSARY_LINE_RE = re.compile(
    r"([A-Z][A-Za-z0-9&/.,'\- ]{1,50}?)\s*[:–—-]\s+"
    r"(.+?)(?=(?:[A-Z][A-Za-z0-9&/.,'\- ]{1,50}?\s*[:–—-]\s+)|$)"
)


def _clean_definition(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(".")


def extract_inline_entries(paragraph: ParagraphUnit, doc_name: str) -> list[GlossaryEntry]:
    entries: list[GlossaryEntry] = []
    text = paragraph.text

    for m in QUOTED_DEFINES_RE.finditer(text):
        entries.append(GlossaryEntry(
            term=m.group(1).strip(), definition=_clean_definition(m.group(2)),
            doc_name=doc_name, page_num=paragraph.page_num, section=paragraph.section_title,
            pattern_type="quoted_defines",
        ))

    for m in ABBR_MEANS_RE.finditer(text):
        entries.append(GlossaryEntry(
            term=m.group(1).strip(), definition=_clean_definition(m.group(2)),
            doc_name=doc_name, page_num=paragraph.page_num, section=paragraph.section_title,
            pattern_type="abbr_means",
        ))

    for m in PAREN_ABBR_RE.finditer(text):
        long_form, abbr = m.group(1).strip(), m.group(2).strip()
        if long_form.lower() == abbr.lower():
            continue
        entries.append(GlossaryEntry(
            term=abbr, definition=_clean_definition(long_form),
            doc_name=doc_name, page_num=paragraph.page_num, section=paragraph.section_title,
            pattern_type="paren_abbr",
        ))

    return entries


def extract_glossary_section_entries(paragraph: ParagraphUnit, doc_name: str) -> list[GlossaryEntry]:
    if not paragraph.section_title or not GLOSSARY_SECTION_RE.search(paragraph.section_title):
        return []
    entries = []
    for m in GLOSSARY_LINE_RE.finditer(paragraph.text):
        term, definition = m.group(1).strip(), _clean_definition(m.group(2))
        if term and definition:
            entries.append(GlossaryEntry(
                term=term, definition=definition, doc_name=doc_name,
                page_num=paragraph.page_num, section=paragraph.section_title,
                pattern_type="glossary_section_line",
            ))
    return entries


def extract_all_entries(paragraphs: list[ParagraphUnit], doc_name: str) -> list[GlossaryEntry]:
    entries: list[GlossaryEntry] = []
    for para in paragraphs:
        entries.extend(extract_glossary_section_entries(para, doc_name))
        entries.extend(extract_inline_entries(para, doc_name))
    return dedupe_entries(entries)


def dedupe_entries(entries: list[GlossaryEntry]) -> list[GlossaryEntry]:
    seen: dict[tuple[str, str], GlossaryEntry] = {}
    for e in entries:
        key = (e.term.strip().lower(), e.definition.strip().lower())
        if key not in seen:
            seen[key] = e
    return list(seen.values())


def build_global_rollup(all_doc_entries: list[list[GlossaryEntry]]) -> list[dict]:
    """Group identical normalized terms across the whole corpus for a fast global lookup."""
    by_term: dict[str, list[GlossaryEntry]] = {}
    for entries in all_doc_entries:
        for e in entries:
            by_term.setdefault(e.term.strip().lower(), []).append(e)

    rollup = []
    for term, entries in sorted(by_term.items()):
        definitions = sorted({e.definition for e in entries})
        rollup.append({
            "term": term,
            "definitions": definitions,
            "doc_count": len({e.doc_name for e in entries}),
            "example_doc": entries[0].doc_name,
        })
    return rollup
