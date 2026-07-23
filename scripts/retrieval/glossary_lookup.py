"""Per-document glossary lookup: expands abbreviations found in retrieved chunk
text using that document's own glossary file only — not the global rollup, since
the same abbreviation can mean something different in another company's filing
(see RETRIEVAL_PLAN.md)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class GlossaryLookup:
    def __init__(self, glossary_dir: Path):
        self.glossary_dir = glossary_dir
        self._doc_cache: dict[str, list[dict[str, Any]]] = {}

    def _load_doc(self, doc_name: str) -> list[dict[str, Any]]:
        if doc_name not in self._doc_cache:
            path = self.glossary_dir / f"{doc_name}.glossary.jsonl"
            entries: list[dict[str, Any]] = []
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
            # Longest term first, so e.g. "MD&A" is checked before a shorter
            # substring that might otherwise match first.
            entries.sort(key=lambda e: len(e["term"]), reverse=True)
            self._doc_cache[doc_name] = entries
        return self._doc_cache[doc_name]

    def expand_terms(self, text: str, doc_name: str) -> list[dict[str, Any]]:
        matches = []
        seen_terms: set[str] = set()
        for entry in self._load_doc(doc_name):
            term = entry["term"]
            if term in seen_terms:
                continue
            if re.search(r"\b" + re.escape(term) + r"\b", text):
                matches.append(entry)
                seen_terms.add(term)
        return matches
