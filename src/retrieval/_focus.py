"""Answer-focused excerpting -- shared by EVERY corpus backend (Wikipedia API, BM25, FAISS body).

A long article body the small model cannot read in full inside the 30s wall; and the answer sentence
usually sits MID-article, not in the lead. So from the option terms (proper-ish words + numbers) we mark
where each appears in the body and keep the LEAD plus a bounded window around each hit -- the topic anchor
and the answer evidence, both, in a few hundred chars. RAW article text throughout (D-008): we excerpt,
never paraphrase.

Born in `WikipediaRetriever` (the Entertainment retrieval overhaul), extracted here so the BM25 local-dump
retriever and the Wikipedia live retriever apply the SAME windowing -- one place to tune it, no drift.
"""
from __future__ import annotations

import re

from schemas import Question

# Words too generic to anchor a body window on -- option boilerplate / question scaffold, these are.
FOCUS_STOP: frozenset[str] = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "their", "they",
    "was", "were", "are", "his", "her", "its", "which", "what", "who", "when", "where",
    "film", "films", "movie", "album", "song", "band", "show", "series", "none", "both",
    "all", "above", "following", "other", "than", "more", "most", "first", "best",
})


def option_terms(question: Question) -> list[str]:
    """The DISTINCTIVE tokens of the options (proper-ish words >=4 chars + numbers) -- the body
    windows we anchor on these. Generic scaffold words (`FOCUS_STOP`) dropped they are."""
    terms: set[str] = set()
    for val in (question.options or {}).values():
        s = str(val)
        for num in re.findall(r"\d[\d.,/]*\d|\d", s):   # "0.7", "360", "1967", a bare "3".
            terms.add(num)
        for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", s):
            if w.lower() not in FOCUS_STOP:
                terms.add(w)
    return [t for t in terms if len(t) >= 2]


def focus(
    body: str,
    terms: list[str],
    lead: str = "",
    chars_focus: int = 1100,
    focus_window: int = 180,
    chars_lead: int = 700,
) -> str:
    """Full body -> lead + bounded windows around where the option terms/numbers appear.

    Keep the lead (topic anchor) plus a ~`focus_window`-char window around each option-term hit, merge
    overlaps, dedupe against the lead, cap the whole at `chars_focus`. No hit -> the lead alone (or the
    body head when no lead given). RAW article text throughout.
    """
    body = re.sub(r"\s+", " ", body or "").strip()
    lead_keep = (lead or "").strip()[:chars_lead]
    if not terms or not body:
        return lead_keep or body[:chars_focus]

    low = body.lower()
    spans: list[tuple[int, int]] = []
    for term in terms:
        for m in re.finditer(re.escape(term.lower()), low):
            spans.append((max(0, m.start() - focus_window),
                          min(len(body), m.end() + focus_window)))
    if not spans:
        return lead_keep or body[:chars_focus]

    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    parts: list[str] = [lead_keep] if lead_keep else []
    used = len(lead_keep)
    lead_low = lead_keep.lower()
    lead_n = len(lead_keep)
    for s, e in merged:
        if used >= chars_focus:
            break
        s = max(s, lead_n)    # trim the part already in the lead -- keep only the post-lead remainder.
        if s >= e:            # the whole window sat inside the lead -- nothing new to add.
            continue
        frag = body[s:e].strip()
        if not frag or frag.lower() in lead_low:   # already covered by the lead.
            continue
        if used + len(frag) > chars_focus:
            frag = frag[:max(0, chars_focus - used)]
        parts.append("… " + frag)
        used += len(frag)
    return " ".join(parts)
