"""Light text utilities for FOMC statements."""

from __future__ import annotations

import re

# Abbreviations that end with a period but don't end a sentence.
_ABBREV = ["U.S.", "Mr.", "Ms.", "Dr.", "Jr.", "St.", "e.g.", "i.e.", "a.m.", "p.m.", "N.W.", "No.", "vs."]
# Split after sentence punctuation, but not after a middle initial such as "Jerome H. Powell".
_SPLIT = re.compile(r"(?<=[.!?])(?<! [A-Z]\.)\s+(?=[A-Z\"(])")


def split_sentences(text: str) -> list[str]:
    """Split a statement into sentences, keeping common abbreviations intact."""
    protected = text
    for i, abbr in enumerate(_ABBREV):
        protected = protected.replace(abbr, f"\x00{i}\x00")
    out = []
    for para in protected.split("\n"):
        for s in _SPLIT.split(para.strip()):
            for i, abbr in enumerate(_ABBREV):
                s = s.replace(f"\x00{i}\x00", abbr)
            s = s.strip()
            if len(s) > 2:
                out.append(s)
    return out


_VOTE = re.compile(r"^(voting (for|against)|in taking the discount rate action|in a related action)", re.IGNORECASE)


def policy_sentences(text: str) -> list[str]:
    """Sentences that carry policy content (drops the voting record and discount rate notes)."""
    return [s for s in split_sentences(text) if not _VOTE.search(s)]
