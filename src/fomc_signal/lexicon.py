"""Dictionary baseline for hawkish/dovish tone.

Each sentence is scored by counting hawkish and dovish phrases. A phrase
preceded by a negation within a few words flips sign ("no longer
accommodative" is hawkish). The dictionaries are hand-built from the
language the FOMC actually uses, in the spirit of Apel and Blix Grimaldi
(2012) but extended for post-2008 balance sheet and guidance wording.

Sentence label: hawkish if score > 0, dovish if < 0, else neutral.
Document score: (hawkish - dovish) / (hawkish + dovish + neutral), in [-1, 1].
"""

from __future__ import annotations

import re

HAWKISH = [
    r"rais(e|ed|ing) the target", r"increase(d)? the target", r"firming", r"tighten(ing|ed)?",
    r"inflation (pressures|risks?) (remain|have|has|are)", r"upside risks? to inflation",
    r"inflation (remains|has remained) elevated", r"elevated inflation", r"higher inflation",
    r"inflation (has )?(picked up|increased|rose|risen|moved up)", r"strong(er)? (growth|demand|labor market)",
    r"robust", r"solid pace", r"labor market (has )?(remained )?tight", r"wage pressures",
    r"further (policy )?firming", r"additional (policy )?firming", r"ongoing increases",
    r"reduc(e|ing) (its|the) holdings", r"balance sheet reduction", r"remove (policy )?accommodation",
    r"less accommodative", r"overheat", r"price pressures", r"resource utilization (is|remains) high",
    r"strongly committed to returning inflation", r"upside risks?", r"upward pressure", r"rising inflation",
    r"inflation (pressures|expectations) .*(rise|rising|increase|elevated|high)", r"restor(e|ing) price stability",
    r"exceed(ed|s)? (the )?(growth|expansion) (in|of) potential", r"taut", r"tight(er|ness)? (labor|product) markets?",
    r"marked up .*inflation", r"higher (energy )?prices", r"pass-through", r"decline in the unemployment",
    r"(growth|activity) (picks|picked) up", r"strength(ened|ening)?", r"expan(d|ded|ding|sion) .*(solid|strong|brisk)",
    r"historically high", r"capacity constraints", r"supply constraints",
]

DOVISH = [
    r"lower(ed|ing)? the target", r"reduc(e|ed|ing) the target", r"\bcuts?\b", r"eas(e|ed|ing)",
    r"accommodat(ive|ion)", r"downside risks?", r"weak(en|ened|ening|ness|er)?", r"slack",
    r"subdued", r"soft(en|ened|ening|er)?", r"slow(ed|ing|er)? (growth|pace|economic)", r"deteriorat",
    r"contract(ion|ed)", r"unemployment (rate )?(has )?(risen|rose|increased|moved up|is elevated|remains elevated)",
    r"patient", r"exceptionally low", r"near zero", r"asset purchases", r"increase (its|the) holdings",
    r"purchase (additional|longer-term)", r"strains", r"turmoil", r"uncertain(ty)? .*(elevated|increased)",
    r"inflation (has )?(declined|eased|moderated|fallen|run below)", r"below (its|the committee's) .*objective",
    r"low inflation", r"disinflation", r"support (the )?economic",
]

_NEG = re.compile(r"\b(no longer|not|less|without|neither|nor)\b(\W+\w+){0,2}\W*$", re.IGNORECASE)
_H = [re.compile(p, re.IGNORECASE) for p in HAWKISH]
_D = [re.compile(p, re.IGNORECASE) for p in DOVISH]


def _count(sentence: str, patterns) -> int:
    n = 0
    for pat in patterns:
        for m in pat.finditer(sentence):
            negated = _NEG.search(sentence[: m.start()]) is not None
            n += -1 if negated else 1
    return n


def sentence_score(sentence: str) -> int:
    return _count(sentence, _H) - _count(sentence, _D)


def classify(sentence: str) -> str:
    s = sentence_score(sentence)
    return "hawkish" if s > 0 else "dovish" if s < 0 else "neutral"


def document_score(sentences: list[str]) -> float:
    labels = [classify(s) for s in sentences]
    if not labels:
        return 0.0
    return (labels.count("hawkish") - labels.count("dovish")) / len(labels)
