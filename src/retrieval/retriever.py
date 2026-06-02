"""RAG retrieval. RAW evidence only this returns -- never a generated answer (a hard rule, D-008 it is).

Three backends, this module gives, and one facade that routes between them:
  * WikipediaRetriever   -- live MediaWiki API; free, no key, RAW extracts. Knowledge topics, it serves.
  * WebSearchRetriever   -- live DuckDuckGo search; RAW result snippets. Post-cutoff NEWS, it serves.
  * FaissRetriever       -- local corpus (Simple Wikipedia), dense vectors. Course-aligned RAG, it is.
  * Retriever (facade)   -- per QUESTION it routes: News -> web, else -> dense/wikipedia.

Why per-question routing, not per-competition? One pipeline ALL six games plays (`run_all_competitions`),
so a single Retriever instance every topic must serve. The News questions a very distinctive shape have
("According to the article published on 2026-05-..", a Guardian byline, an ISO date) -- on that we route,
far more reliably than the generic topic classifier (which `adapt_question` leaves unset in live play).

The retrieved text feeds the prompt as RAW context (`prompting.builder._build_context_block`); the LLM
still reasons over it, it does. No backend ever an answer generates -- raw chunks only, always.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from schemas import Question, RetrievedDoc
from retrieval.wikipedia import WikipediaRetriever   # the polished live backend, reuse it we do.


# --------------------------------------------------------------------------- #
# News detection -- THIS game's News questions a tell-tale signature carry.
# --------------------------------------------------------------------------- #

# An ISO date (2026-05-15), an "according to .. article" lead-in, a "published on", a Guardian byline --
# any one of these, a News question it betrays. On the offline mix AND live play alike, the text the same is.
_NEWS_SIGNATURE = re.compile(
    r"\b20\d{2}-\d{2}-\d{2}\b"               # An ISO date, the strongest tell it is.
    r"|according\s+to\s+(?:the|a|an)\b.*\barticle\b"
    r"|\bpublished\s+on\b"
    r"|\bthe\s+guardian\b",
    re.IGNORECASE | re.DOTALL,
)


def _looks_like_news(question: Question) -> bool:
    """True when a post-cutoff NEWS question this is -- to the live web, route it we should."""
    # The topic, when the caller set it (offline dataset / a future client enrichment), trust we do.
    if question.topic and "news" in question.topic.lower():
        return True
    # Else the text's own signature, read we do -- reliable for this game's News, it is.
    return bool(_NEWS_SIGNATURE.search(question.text or ""))


def _query_from_question(question: Question, max_chars: int = 300) -> str:
    """A clean search query, from the question text distil it we do.

    The "According to the article published on <date>," boilerplate, strip it we do -- noise for a
    search engine it is, the real entities it buries. Whitespace collapsed, length capped, the rest is.
    """
    text = (question.text or "").strip()
    # The dated attribution LEAD-IN clause (up to the first comma), drop it we do -- pure search noise it
    # is. MANY shapes this game uses, and only "according to.." we caught before -- "In the report from
    # 2026-05-17, .." slipped through and the date killed the search (qid 11503 -> 0 hits). So broadened:
    text = re.sub(
        r"^\s*(?:"
        r"according\s+to\b[^,]*,"                  # According to the 2026-.. report, ..
        r"|in\s+the\b[^,]*\b(?:report|article)\b[^,]*,"   # In the (news) report from/published .., ..
        r"|as\s+reported\b[^,]*,"                  # As reported on 2026-.., ..
        r"|on\s+20\d{2}-\d{2}-\d{2}\b[^,]*,"       # On 2026-05-06, ..
        r")\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # EMBEDDED attribution + bare dates anywhere, strip too -- "..on 2026-05-16 according to the article,.."
    # (qid 11725) the date mid-sentence pins nothing useful; the date WINDOW handles temporality separately.
    text = re.sub(r",?\s*according\s+to\s+the\s+(?:news\s+)?(?:article|report)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:on|from|published\s+on|dated)\s+20\d{2}-\d{2}-\d{2}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b20\d{2}-\d{2}-\d{2}\b", "", text, flags=re.IGNORECASE)
    # Whitespace, collapse it we do; the length, cap it we must (a search box, finite it is).
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


_ISO_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


def _date_range(question: Question, before_days: int = 3, after_days: int = 2):
    """(lo_iso, hi_iso) around the question's ISO date, or (None, None). Shared by gnews + Guardian.

    News questions a date carry ("the article from 2026-05-14"). As free text it is NOISE (drags the
    query off-topic), but as a DATE RANGE it culls the temporally-irrelevant -- so the date we lift out
    and re-inject as a range operator (Google `after:/before:`, Guardian `from-date/to-date`)."""
    m = _ISO_DATE.search(question.text or "")
    if not m:
        return None, None
    try:
        from datetime import date, timedelta
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return (d - timedelta(days=after_days)).isoformat(), (d + timedelta(days=before_days)).isoformat()
    except Exception:
        return None, None


def _gnews_date_window(question: Question) -> str:
    """The Google-News ` after:.. before:..` operator from the question's date (else "")."""
    lo, hi = _date_range(question)
    return f" after:{lo} before:{hi}" if lo else ""


def _question_date(question: Question):
    """The question's ISO date as a `datetime.date` (else None) -- the anchor for recency re-ranking."""
    m = _ISO_DATE.search(question.text or "")
    if not m:
        return None
    try:
        from datetime import date
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


# A gnews hit this many days from the question's date we treat as STALE -- a 2019 article for a 2026
# question is the answer to a different event. Soft, not hard: dropped only when fresher hits remain.
_STALE_DAYS = 365


def _rerank_by_recency(items: list, question: Question) -> list:
    """Re-order gnews items so the ones published NEAR the question's date come first -- the precision
    fix for "the article is retrievable but a STALE one out-ranks it" (qid 12133: a 2019 Rosalía piece
    out-ranked the 2026-05-06 article). Items are `(text, link, pub_date)`.

    - DROP clearly-stale items (>_STALE_DAYS from the question date) -- but only if fresher ones remain,
      so we never empty the list chasing recency.
    - Then STABLE-sort by month-proximity, preserving the original (relevance) order within a month --
      so recency refines the ranking without discarding gnews's relevance signal.
    No question date (or no dated items) -> unchanged (we don't invent an order)."""
    qd = _question_date(question)
    if qd is None or not items:
        return items
    def days(it):
        d = it[2] if len(it) > 2 else None
        return abs((d - qd).days) if d is not None else None
    fresh = [it for it in items if (days(it) is not None and days(it) <= _STALE_DAYS)]
    if not fresh:
        return items   # nothing dated-and-fresh -> don't second-guess (keep recall, e.g. a Jan event a May report cites).
    undated = [it for it in items if days(it) is None]
    # month-bucket so same-month hits keep their relevance order; closer months first; undated kept last-but-present.
    ranked = sorted(fresh, key=lambda it: days(it) // 30)
    return ranked + undated


# --------------------------------------------------------------------------- #
# Option-aware body focusing -- the attribution/detail fix.
# --------------------------------------------------------------------------- #
# How much RAW article text the body fetchers harvest BEFORE we focus-select. Bigger than the per-doc
# prompt budget on purpose: the answer sentence ("Prof X said..") often sits mid-article, so we pull a
# lot, then keep only the relevant windows. ~3-4s/fetch unchanged -- this is post-fetch string work.
_RAW_BODY_CHARS = 3500

# LATENCY GUARD for the (slow) browser body path. The relevance filter routes ~44% of News turns to the
# headless browser, and each body fetch is ~3-4s -- a 10-round News test showed browser turns at a 19.8s
# median (vs 10.9s on Guardian) and ONE 31s turn that the server timed out -> a level-0 death (0 points).
# A timeout is worse than a wrong guess, so we cap the browser stage HARD: at most this many bodies, and a
# wall-clock budget after which we stop fetching (the headlines still carry the turn). Guardian (API, ~0.2s)
# is NOT capped here -- only the browser is the latency risk.
_MAX_BROWSER_BODIES = 2
_BROWSER_BUDGET_S = 12.0

# A run of Capitalised words -- a proper name / place an MCQ option carries ("Naveed Sattar", "Red Sea").
_PROPER_SPAN = re.compile(r"\b[A-Z][a-zA-Z.'’-]+(?:\s+[A-Z][a-zA-Z.'’-]+)*")

# Function words an option may carry -- never worth a body window on their own ("the war", "a group").
_OPT_STOPWORDS = frozenset(
    "the a an of to in on and or for with by at as from that this it its their his her "
    "was were is are be been being had has have will would could should".split()
)


def _stem(w: str) -> str:
    """A crude suffix-strip so an option word matches its article-text variants -- 'glaciers'/'glacial'
    -> 'glaci', 'volcanic' -> 'volcan', 'melting' -> 'melt'. No linguistic claim; recall, the goal is."""
    w = w.lower()
    for suf in ("ation", "ings", "ing", "ers", "er", "ed", "es", "ic", "al", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _option_patterns(question: Question) -> list[str]:
    """Regex alternatives that LOCATE, in an article body, the text an MCQ option refers to. For each
    option value: its full phrase (verbatim), its Capitalised proper-name spans ("Naveed Sattar"), AND
    the STEM of each content word -- so "Melting glaciers" finds "glacial"/"glacier thinning", not only
    the exact phrase. The generic-phrase miss (qid 10630) this fixes.

    NB: we use these to SELECT which slice of an ALREADY-RETRIEVED body to keep -- NOT to build the
    search query (query-side option injection was a dead end: it dragged the search off-topic)."""
    pats: list[str] = []
    for v in (question.options or {}).values():
        v = re.sub(r"\s+", " ", (v or "")).strip()
        if len(v) < 3:
            continue
        pats.append(re.escape(v.lower()))                          # the full phrase, verbatim.
        for m in _PROPER_SPAN.findall(v):                          # proper names / places.
            if len(m) >= 4:
                pats.append(re.escape(m.lower()))
        for w in re.findall(r"[A-Za-z]+", v):                      # content-word stems (prefix match).
            if len(w) >= 4 and w.lower() not in _OPT_STOPWORDS:
                pats.append(r"\b" + re.escape(_stem(w)) + r"\w*")
        for num in re.findall(r"\d[\d,]*(?:\.\d+)?", v):           # NUMBERS ("70%", "1.4m") -- so a
            # percentage/quantity question windows on the article's STATISTIC sentence, not on a stray
            # word. Commas dropped so "15,300" also finds a "15300" in the text (qid 11782: the % miss).
            pats.append(re.escape(num))
            if "," in num:
                pats.append(re.escape(num.replace(",", "")))
    seen, out = set(), []
    for p in pats:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _focus_body(body: str, question: Question, char_limit: int,
                window: int = 260, head_chars: int = 280) -> str:
    """An option-aware slice of an article body, capped at `char_limit`.

    Attribution/detail News questions ("which expert said..", "what caused..") hinge on a sentence that
    often sits MID-article -- a plain head-truncation to char_limit drops it. So when the options give
    us search terms (a name, a place, a cause), we KEEP the windows of text around where those terms
    (or their stems) appear, plus the lead (topic/setup). No option term lands in this article ->
    head-truncation, the old safe behaviour (no regression for options not echoed in the text)."""
    body = re.sub(r"\s+", " ", body or "").strip()
    if len(body) <= char_limit:
        return body
    spans: list[list[int]] = [[0, head_chars]]   # the lead, always kept (context for the windows).
    for pat in _option_patterns(question):
        try:
            for m in re.finditer(pat, body, re.IGNORECASE):
                spans.append([max(0, m.start() - window), min(len(body), m.end() + window)])
        except re.error:
            continue
    if len(spans) == 1:                 # only the lead -> no option term found -> head truncation.
        return body[:char_limit]
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:                  # adjacent/overlapping windows, fuse them we do.
        if merged and s <= merged[-1][1] + 40:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, total = [], 0
    for s, e in merged:
        chunk = body[s:e].strip()
        if total + len(chunk) > char_limit:
            chunk = chunk[: max(0, char_limit - total)]
        if chunk:
            out.append(chunk)
            total += len(chunk)
        if total >= char_limit:
            break
    return " … ".join(out).strip()


# --------------------------------------------------------------------------- #
# Guardian relevance gate -- "is this body actually about the question?"
# --------------------------------------------------------------------------- #
# Generic / question-scaffold words that carry NO topic signal -- excluded from the keyword test, so a
# stray "potentially" or "America" in an off-topic article cannot fake relevance. The DISTINCTIVE nouns
# (measles, resurgence, a proper name) are what must land for a body to count as on-topic.
_Q_STOPWORDS = frozenset((
    "the a an of to in on and or for with by at as from that this it its their his her was were is are be "
    "been being had has have will would could should which what who whom whose where when why how than "
    "according article articles report reported reports published said say says mention mentioned noted "
    "note stated state describe described decided decide launch launched potentially particularly "
    "especially recently currently generally about into them they you your our we us not no yes also more "
    "most some any such event events people person government country countries world year years day days "
    "time news north south east west america american europe european asia asian africa african region "
    "regarding consequence following between against during while because"
).split())


def _question_keywords(question: Question) -> list[str]:
    """The DISTINCTIVE topic terms of a question -- its proper names + its longer content words (>=6),
    minus the generic scaffolding. The handful that an on-topic article MUST mention."""
    text = question.text or ""
    kws: list[str] = []
    for part in _PROPER_SPAN.findall(text):          # proper names: the strongest topic signal.
        for w in part.split():
            wl = w.lower()
            if len(wl) >= 4 and wl not in _Q_STOPWORDS:
                kws.append(wl)
    for w in re.findall(r"[A-Za-z]+", text):          # long content words.
        wl = w.lower()
        if len(wl) >= 6 and wl not in _Q_STOPWORDS:
            kws.append(wl)
    seen, out = set(), []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# Witness/attribution verbs -- a question's "..that <person> EXPERIENCED" clause pins a body WITNESS, not
# the searchable event. Live evidence: "..explosion that Ángel Linares and his neighbors experienced" -> 0
# gnews hits, but the bare event "initial explosion" (+ option terms) found the Caracas-strike article.
_WITNESS_VERB = (
    r"(?:experienced|witnessed|described|mentioned|mention|saw|heard|felt|reported|recalled|observed"
    r"|noticed|encountered|said|noted|claimed|stated|revealed|faced)"
)


def _strip_body_details(text: str) -> str:
    """Drop the body-WITNESS noise from a question so the EVENT is what we search for. Two shapes:
      * a relative clause  "<event> that <person ...> experienced"  -> keep "<event>", drop the clause;
      * an attribution     "what X did <person ...> mention"        -> drop the "did <person> mention" span.
    The witness name is article-body detail that drags a news search off-topic; the event is what's
    indexed. Used ONLY for the option-augmented FALLBACK re-search (never the primary query)."""
    text = re.sub(rf"\bthat\s+[^?.,]{{1,55}}?\s+{_WITNESS_VERB}\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(rf"\bdid\s+[^?.,]{{1,45}}?\s+{_WITNESS_VERB}\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" ?,.")


# Question-MECHANICS words -- scaffolding that names HOW the question is asked, not WHAT it's about.
# Google News RSS is AND-like: ANY term not in the article title/desc zeroes the result set (verified --
# "Mike Smith fighter pilot climate activist" = 1 hit, but "+influenced +transition" = 0). So for the
# keyword query these MUST go, leaving only the topical entities/nouns the indexed headline will carry.
_SCAFFOLD = frozenset((
    "event influenced transition reveals reveal revealed example action significant regarding outcome "
    "expected strategy reason benefit primary context mentioned mention described describe describes "
    "according report article published take took used using called caused occurred involving experienced "
    "identified focused response decision considered following participation discrepancies discrepancy "
    "initial main major specific aspect type kind number amount percentage country town highlighted "
    "noted point contention issue measure"
).split())


def _keyword_query(question: Question, max_proper: int = 2, max_or: int = 6) -> str:
    """A gnews query of the form  "<entity>" "<entity>" (word OR word OR ..)  -- the proper-name entities
    REQUIRED (quoted), the descriptive content words OR'd. Two failure modes of a plain query this fixes:
      * a full SENTENCE is AND-matched by gnews -> a noise word ("influenced", "transition") zeroes the
        result set even though the article is indexed (qids 11917/11910); OR'ing the content words means
        the headline need only carry SOME of them, so the article surfaces;
      * ANDing every term over-constrains, but ORing EVERYTHING is too loose -> requiring the proper-name
        entities keeps it on-topic (qid 12133: require "Rosalia", OR the rest -> her articles, not generic
        "Latin America" finance pieces).
    Scaffolding ("event/report/reason/..") and witness clauses are stripped first. Used as a RETRY."""
    text = _strip_body_details(_query_from_question(question))
    proper = sorted({m for m in _PROPER_SPAN.findall(text) if len(m) >= 3}, key=len, reverse=True)[:max_proper]
    proper_words = {w.lower() for m in proper for w in m.split()}
    content, seen = [], set()
    for w in re.findall(r"[A-Za-z]+", text):
        wl = w.lower()
        if (len(wl) >= 4 and wl not in _Q_STOPWORDS and wl not in _OPT_STOPWORDS
                and wl not in _SCAFFOLD and wl not in proper_words and wl not in seen):
            seen.add(wl)
            content.append(w)
    content = content[:max_or]
    required = " ".join(f'"{p}"' for p in proper)
    or_group = "(" + " OR ".join(content) + ")" if content else ""
    return (required + " " + or_group).strip()


def _option_query_terms(question: Question, max_terms: int = 6) -> str:
    """The distinctive words of the MCQ OPTIONS, as a space-joined search fragment -- to APPEND to a
    gnews query when the base results look off-topic. Live evidence: a vague question ("what caused the
    explosion..") missed the answer article, but "+missiles" / "+CEPI" (option terms) surfaced it.

    Proper names + content words (>=4, non-stopword) across all options, deduped, longest-first, capped.
    NOT used in the FIRST search (options-in-query can dilute an already-good query) -- only the fallback."""
    terms: list[str] = []
    for v in (question.options or {}).values():
        v = re.sub(r"\s+", " ", (v or "")).strip()
        for m in _PROPER_SPAN.findall(v):
            for w in m.split():
                if len(w) >= 4 and w.lower() not in _OPT_STOPWORDS:
                    terms.append(w)
        for w in re.findall(r"[A-Za-z]+", v):
            if len(w) >= 4 and w.lower() not in _OPT_STOPWORDS:
                terms.append(w)
    seen, out = set(), []
    for t in sorted(terms, key=len, reverse=True):
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            out.append(t)
    return " ".join(out[:max_terms])


def _headlines_on_topic(items: list, question: Question) -> bool:
    """True when at least one gnews headline shares a distinctive keyword with the question -- the cheap
    "did the base search find anything relevant?" test that gates the option-augmented re-search."""
    keywords = _question_keywords(question)
    if not keywords:
        return True   # nothing distinctive to test -> don't second-guess the search.
    return any(_relevance(t, keywords) > 0 for t, *_rest in items)


def _relevance(text: str, keywords: list[str]) -> float:
    """Fraction of the question's distinctive keywords whose stem appears in `text`. 1.0 when there are
    no keywords to test (then we don't second-guess the search)."""
    if not keywords:
        return 1.0
    low = (text or "").lower()
    hits = 0
    for k in keywords:
        try:
            if re.search(r"\b" + re.escape(_stem(k)) + r"\w*", low):
                hits += 1
        except re.error:
            continue
    return hits / len(keywords)


# The Guardian-keep bar: a body must share at least this fraction of the question's distinctive keywords,
# else we judge it OFF-TOPIC and fall through to the broad web path (gnews links via browser/ddg). Modest
# (a third) so a genuinely on-topic Guardian article -- today's 9/10 -- is never abandoned.
_GUARDIAN_RELEVANCE_MIN = 0.34


# --------------------------------------------------------------------------- #
# Backend 1 -- live Wikipedia: reused from `retrieval.wikipedia.WikipediaRetriever`.
# Entity-first search, a 429 retry, a shared session -- already polished it is, so duplicate it we do not.
# (Imported at the top.) Its knobs: top_k, lang, timeout, chars_per_doc, search_limit.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Backend 2 -- live web search (DuckDuckGo HTML; for post-cutoff NEWS).
# --------------------------------------------------------------------------- #

# A result snippet in DDG's HTML lite endpoint, this matches -- the <a class="result__snippet">..</a> text.
_DDG_SNIPPET = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")  # Inner HTML tags (bold highlights), strip them we do.


def _unescape_html(s: str) -> str:
    import html as _html
    return _html.unescape(s)


def _is_prose(t: str) -> bool:
    """True when a paragraph REAL article prose looks like -- not nav/menu/cookie/JS boilerplate.

    The tells of body text: long enough, a sentence end it has, and real word-spacing (nav menus mash
    CamelCase with few spaces; JS blobs carry `window.`/`{`). Crude but it keeps the article, drops the chrome.
    """
    if len(t) < 50 or len(t.split()) < 10:
        return False
    if "window." in t or "function(" in t or t[:30].count("{"):
        return False
    if ". " not in t and not t.endswith("."):
        return False
    return (t.count(" ") / len(t)) >= 0.12


class WebSearchRetriever:
    """query -> top-k RAW web result snippets. Google News RSS first, DuckDuckGo HTML second.

    For NEWS only this is -- the post-cutoff events Wikipedia cannot know (a Malian minister killed on a
    2026 date, a whale named Timmy). RAW snippets/headlines we return, the rule honouring -- no answer we
    synthesise. Google News RSS the default primary it is (keyless, raw, reliable on Colab where the DDG
    scrape gets blocked); NAME it in the video, you must.

    Brittle, web scraping inherently is (a layout change, a bot block, a 429). So crash-safe entirely it
    stays: empty list on ANY failure. A `search_fn` injection point we expose -- a different free source
    (a news RSS, a search API you name in the video) drop in here you can, without touching the routing.
    """

    _URL = "https://html.duckduckgo.com/html/"
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    }

    def __init__(
        self,
        top_k: int = 3,
        char_limit: int = 400,
        timeout_s: float = 6.0,
        search_fn: Optional[Callable[[str, int], list[RetrievedDoc]]] = None,
        fetch_bodies: int = 0,
        body_mode: str = "ddg",
        guardian_api_key: str = "",
    ):
        self.top_k = top_k
        self.char_limit = char_limit
        self.timeout_s = timeout_s
        # An override hook -- when given, OURS it replaces (a Guardian RSS, a NewsAPI, your choice).
        self._search_fn = search_fn
        # How many of the TOP RSS articles to also fetch the body of (0 = headlines only), and HOW.
        self.fetch_bodies = max(0, int(fetch_bodies))
        self.body_mode = (body_mode or "ddg").lower()
        # The Guardian Open Platform key -- when present, the FAST primary body source it is (env fallback).
        self.guardian_api_key = guardian_api_key or os.environ.get("GUARDIAN_API_KEY", "")

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        query = _query_from_question(question)
        if not query:
            return []
        # An injected source (the video-named API), if given -- OURS it replaces entirely.
        if self._search_fn is not None:
            try:
                return self._search_fn(query, self.top_k)
            except Exception:
                return []
        # Default News stack: Google News RSS for HEADLINES -- keyless, raw RSS, reliable on the Colab IP.
        # The post-cutoff answer is OFTEN in the headline itself ("...lists 41 properties.. - BBC").
        # The question's ISO date, as a `after:.. before:..` window we re-inject (Google-only) -- the
        # temporally-irrelevant noise it culls. Over-narrowed (0 hits)? -> without the window, retry once.
        window = _gnews_date_window(question)
        try:
            items = self._gnews_items(query + window)   # [(text, link)]
            if not items and window:
                items = self._gnews_items(query)
        except Exception:
            items = []
        # KEYWORD RE-SEARCH (#4): a full-sentence query is AND-matched by gnews, so a long question often
        # returns 0 even though the answer article IS indexed (qids 11917/11910). When the sentence comes
        # up empty/off-topic, retry with the SHORT entity/noun query -- found articles a sentence missed.
        if not items or not _headlines_on_topic(items, question):
            kwq = _keyword_query(question)
            # >=2 terms required: a lone generic word ("explosion") matches mountains of off-topic news
            # and would pre-empt the sharper option fallback (qid 10813 -> Caracas needs "+missiles").
            if kwq and len(kwq.split()) >= 2 and kwq.lower() != query.lower():
                try:
                    kw_items = self._gnews_items(kwq + window) or self._gnews_items(kwq)
                except Exception:
                    kw_items = []
                if kw_items:
                    items = kw_items + [it for it in items if it not in kw_items]
        # OPTION-AUGMENTED RE-SEARCH (fallback): the base query found nothing on-topic (a too-vague or
        # body-detail-polluted question), so re-search WITH the option keywords appended -- live evidence:
        # "+missiles"/"+CEPI" surfaced the answer article a bare query missed (qids 10813, 10659). Gated on
        # off-topic so a working query is never diluted (the options-in-query dead end). Option-hits FIRST.
        if not _headlines_on_topic(items, question):
            opt_terms = _option_query_terms(question)
            if opt_terms:
                # #2: strip the body-witness clause so the EVENT (not the witness name) drives the search.
                aug = (_strip_body_details(query) + " " + opt_terms).strip()
                try:
                    aug_items = self._gnews_items(aug + window) or self._gnews_items(aug)
                except Exception:
                    aug_items = []
                if aug_items:
                    seen = {t for t, _l, _d in aug_items}
                    items = aug_items + [it for it in items if it[0] not in seen]
        # PRECISION: re-rank by proximity to the question's date -- a stale hit never out-ranks the dated
        # article (qid 12133: a 2019 Rosalía piece). Drops clearly-stale items when fresher ones remain.
        items = _rerank_by_recency(items, question)
        headlines = [
            RetrievedDoc(doc_id=f"gnews:{i}", text=t[: self.char_limit], source="google_news_rss", score=0.0)
            for i, (t, _link, _d) in enumerate(items[: self.top_k])
        ]
        # BODIES (best-effort): "who was quoted.." / exact numbers live in the article TEXT, not the headline.
        #   1) Guardian API FIRST -- raw `bodyText` in ONE ~0.2s call (no browser, no consent wall). Many of
        #      these questions' answer articles ARE the Guardian, so most turns end here, FAST (~4s total).
        #   2) Else (non-Guardian story) -> the broad fallback: browser (Colab) or ddg-direct.
        # Bodies first (richer), headlines after. Any failure -> just the headlines, the turn never sunk.
        bodies: list[RetrievedDoc] = []
        if self.fetch_bodies > 0 and self.body_mode != "off":
            try:
                guardian = self._guardian_bodies(query, question, self.fetch_bodies)
            except Exception:
                guardian = []
            # RELEVANCE FILTER (per doc, not per set): Guardian is tried first (fast), but each body is
            # KEPT only when it actually matches the question. Off-topic Guardian hits (the answer article
            # isn't the Guardian's) used to be returned wholesale -- burying the one on-topic gnews link
            # (the measles/World-Cup miss, qid 11067) AND, worse, DILUTING a good body with junk ones (the
            # WHO-treaty miss, where 2 of 3 Guardian docs were a nude-art review + Eurovision). Now:
            #   * keep the Guardian bodies that are on-topic; if ANY are, use only those;
            #   * if NONE are, fall THROUGH to the broad web path (gnews links via browser/ddg);
            #   * then drop the junk there too (a captcha/Cloudflare block page scores ~0).
            keywords = _question_keywords(question)
            guardian_rel = [d for d in guardian if _relevance(d.text, keywords) >= _GUARDIAN_RELEVANCE_MIN]
            if guardian_rel:
                bodies = guardian_rel
            else:
                try:
                    if self.body_mode == "browser":
                        fetched = self._fetch_bodies_via_browser(
                            [lnk for _t, lnk, _d in items], self.fetch_bodies, question)
                    else:
                        fetched = self._fetch_bodies_via_ddg(query, self.fetch_bodies, question)
                except Exception:
                    fetched = []
                # Drop the individually-junk fetched bodies (block/captcha pages) -- but if that leaves
                # nothing, keep what we got (a weak body beats none). Last resort: Guardian, even off-topic.
                fetched_rel = [d for d in fetched if _relevance(d.text, keywords) >= _GUARDIAN_RELEVANCE_MIN]
                bodies = fetched_rel or fetched or guardian
        docs = bodies + headlines
        if docs:
            return docs
        # Headlines empty too (gnews down) -- DDG snippets a last try; then the router casts to Wikipedia.
        try:
            return self._ddg_search(query)
        except Exception:
            return []

    # -- internals --

    def _gnews_items(self, query: str) -> list[tuple]:
        """Google News RSS -> [(headline_text, article_link, pub_date)]. Keyless raw RSS, rule-compliant.

        The `item/title` a clean "Headline - Publisher" string is -- the recent fact, often IN it. The
        `item/link` the (consent-walled) article URL is -- only the BROWSER body path can open it. The
        `item/pubDate` we now also lift -> a `datetime.date` (or None): the PRECISION signal that lets us
        rank by proximity to the question's date, so a stale 2019 hit never out-ranks the dated article.
        NAME this in the video ("Google News RSS").
        """
        import urllib.parse
        import xml.etree.ElementTree as ET
        from email.utils import parsedate_to_datetime

        url = (
            "https://news.google.com/rss/search?q="
            + urllib.parse.quote(query)
            + "&hl=en-US&gl=US&ceid=US:en"
        )
        resp = requests.get(url, headers=self._HEADERS, timeout=self.timeout_s)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items: list[tuple] = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            desc = _unescape_html(_TAG.sub("", item.findtext("description") or "")).strip()
            # The description often just repeats the title (+ source list) -- append it only when it adds.
            text = title if (not desc or desc == title) else f"{title}. {desc}"
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            try:
                pub = parsedate_to_datetime(item.findtext("pubDate") or "").date()
            except Exception:
                pub = None
            items.append((text, (item.findtext("link") or "").strip(), pub))
        return items

    def _guardian_bodies(self, query: str, question: Question, n: int) -> list[RetrievedDoc]:
        """The Guardian Content API -> top-`n` articles WITH full `bodyText`, in ONE call. [] if no key,
        no match, or any error. Free, raw journalism (no synthesis), ~0.2s -- no browser, no consent wall.

        Only the Guardian's OWN content it covers, but these questions' answer articles often ARE the
        Guardian -> most News turns end here, fast. NAME "Guardian Open Platform API" in the video."""
        if not self.guardian_api_key:
            return []
        params = {
            "q": query,
            "show-fields": "bodyText",
            "order-by": "relevance",
            "page-size": max(1, n),
            "api-key": self.guardian_api_key,
        }
        lo, hi = _date_range(question)
        if lo:
            params["from-date"], params["to-date"] = lo, hi
        resp = requests.get(
            "https://content.guardianapis.com/search",
            params=params, headers=self._HEADERS, timeout=self.timeout_s,
        )
        resp.raise_for_status()
        results = (resp.json().get("response") or {}).get("results") or []
        docs: list[RetrievedDoc] = []
        for i, art in enumerate(results[:n]):
            body = re.sub(r"\s+", " ", ((art.get("fields") or {}).get("bodyText") or "")).strip()
            if body:
                # FULL bodyText we keep, then OPTION-AWARE focus it to the budget -- the attribution
                # sentence (often mid-article) survives, where a head-truncation would have cut it.
                docs.append(RetrievedDoc(
                    doc_id=f"guardian:{i}", text=_focus_body(body, question, self.char_limit),
                    source="theguardian.com", score=0.0,
                ))
        return docs

    def _fetch_bodies_via_browser(self, links: list, n: int, question: Question) -> list[RetrievedDoc]:
        """Headless Chromium opens each Google-News link, runs the JS past the consent wall, reads the
        rendered article. The ONLY body path that works on Colab. [] when Playwright/Chromium absent.

        A LOT of raw prose we harvest (`_RAW_BODY_CHARS`), then OPTION-AWARE focus it to the budget --
        the mid-article attribution survives where the old head-truncation would have cut it."""
        from .browser_fetch import get_browser_fetcher

        fetcher = get_browser_fetcher(nav_timeout_s=min(self.timeout_s + 2.0, 8.0))
        n = min(n, _MAX_BROWSER_BODIES)          # cap COUNT -- the 3rd browser body is what blew the 30s wall.
        start = time.monotonic()
        docs: list[RetrievedDoc] = []
        for i, link in enumerate(links):
            if len(docs) >= n:
                break
            if time.monotonic() - start > _BROWSER_BUDGET_S:   # WALL guard -- stop before the 30s timeout.
                break
            if not link:
                continue
            body = fetcher.fetch(link, max_chars=_RAW_BODY_CHARS)
            if body:
                docs.append(RetrievedDoc(
                    doc_id=f"browser:body:{i}",
                    text=_focus_body(body, question, self.char_limit),
                    source="headless_chromium", score=0.0,
                ))
        return docs

    def _fetch_bodies_via_ddg(self, query: str, n: int, question: Question) -> list[RetrievedDoc]:
        """DuckDuckGo HTML -> the top-`n` DIRECT publisher URLs -> their article BODIES. [] on any failure.

        DDG result links carry the real URL in a `uddg=` redirect param (bbc.co.uk/.., not a Google
        consent wall) -- decode it, fetch it, the prose paragraphs harvest. RAW article text, no synthesis.
        """
        import urllib.parse

        resp = requests.post(
            self._URL, data={"q": query}, headers=self._HEADERS, timeout=self.timeout_s,
        )
        resp.raise_for_status()
        urls: list[str] = []
        for href in re.findall(r'result__a"[^>]*href="([^"]+)"', resp.text):
            m = re.search(r"uddg=([^&]+)", href)
            real = urllib.parse.unquote(m.group(1)) if m else href
            if real.startswith("//"):
                real = "https:" + real
            if real.startswith("http") and "duckduckgo.com" not in real:
                urls.append(real)
            if len(urls) >= n:
                break
        docs: list[RetrievedDoc] = []
        for i, u in enumerate(urls):
            # A LOT of raw prose we harvest, then OPTION-AWARE focus it -- mid-article attribution survives.
            body = self._fetch_article_text(u, max_chars=_RAW_BODY_CHARS)
            if body:
                host = re.sub(r"^https?://(www\.)?", "", u).split("/")[0]
                docs.append(RetrievedDoc(
                    doc_id=f"web:body:{i}",
                    text=_focus_body(body, question, self.char_limit), source=host, score=0.0,
                ))
        return docs

    def _fetch_article_text(self, url: str, max_chars: int = 700) -> str:
        """Best-effort: fetch a DIRECT publisher URL, its PROSE paragraph text return. "" on ANY failure.

        The `<p>..</p>` we harvest, but keep only PROSE -- a paragraph with a sentence end and real
        word-spacing. This drops the nav/menu/cookie boilerplate (mashed CamelCase, `window.WIZ_..` JS)
        that else poisons the context. RAW article text it is, no synthesis. A TIGHT timeout (<=4s) the
        30s wall protects -- a slow site never the turn it sinks.
        """
        if not url:
            return ""
        try:
            resp = requests.get(
                url, headers=self._HEADERS,
                timeout=min(self.timeout_s, 4.0), allow_redirects=True,
            )
            resp.raise_for_status()
            paras: list[str] = []
            total = 0
            for raw in re.findall(r"<p[^>]*>(.*?)</p>", resp.text, re.IGNORECASE | re.DOTALL):
                t = re.sub(r"\s+", " ", _unescape_html(_TAG.sub("", raw))).strip()
                if not _is_prose(t):
                    continue
                paras.append(t)
                total += len(t)
                if total >= max_chars:
                    break
            return " ".join(paras).strip()[:max_chars]
        except Exception:
            return ""

    def _ddg_search(self, query: str) -> list[RetrievedDoc]:
        resp = requests.post(
            self._URL, data={"q": query}, headers=self._HEADERS, timeout=self.timeout_s,
        )
        resp.raise_for_status()
        docs: list[RetrievedDoc] = []
        for i, raw in enumerate(_DDG_SNIPPET.findall(resp.text)):
            text = _unescape_html(_TAG.sub("", raw)).strip()
            if not text:
                continue
            docs.append(RetrievedDoc(
                doc_id=f"ddg:{i}",
                text=text[: self.char_limit],
                source="duckduckgo",
                score=0.0,
            ))
            if len(docs) >= self.top_k:
                break
        return docs


# --------------------------------------------------------------------------- #
# Backend 3 -- local FAISS over a corpus (the course's dense RAG).
# --------------------------------------------------------------------------- #

class FaissRetriever:
    """query -> top-k RAW chunks from a LOCAL corpus. multilingual-e5 + FAISS, this is.

    The index a directory is: `<index_path>/index.faiss` + `<index_path>/docs.jsonl` (one
    `{doc_id, text, source}` per line, row-aligned to the FAISS vectors). Build it with
    `src/retrieval/build_index.py`, you do (on Colab, once per corpus).

    e5 a prefix convention has: PASSAGES "passage: " at index time, QUERIES "query: " at search time.
    Honour it we must, or the cosine scores meaningless they are. Normalised embeddings + inner-product
    index => cosine similarity, this gives. Heavy deps (faiss, sentence-transformers) LAZILY loaded they
    are -- importing this module, a GPU it must never wake.
    """

    def __init__(
        self,
        index_path: str,
        embedder: str = "intfloat/multilingual-e5-small",
        top_k: int = 3,
        char_limit: int = 600,
        min_score: float = 0.0,
    ):
        self.index_path = Path(index_path)
        self.embedder_name = embedder
        self.top_k = top_k
        self.char_limit = char_limit
        self.min_score = min_score   # Cosine floor: docs below it, drop we do (0.0 -> keep all).
        self._model = None     # Lazily loaded, the SentenceTransformer is.
        self._index = None     # Lazily loaded, the FAISS index is.
        self._docs: list[dict] = []

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        idx_file = self.index_path / "index.faiss"
        docs_file = self.index_path / "docs.jsonl"
        if not idx_file.exists() or not docs_file.exists():
            raise FileNotFoundError(
                f"No FAISS index at {self.index_path} -- build it with build_index.py, you must."
            )
        self._index = faiss.read_index(str(idx_file))
        self._docs = [
            json.loads(line)
            for line in docs_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._model = SentenceTransformer(self.embedder_name)

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        try:
            self._ensure_loaded()
        except Exception:
            # No index / a load failure -- unaided the model answers. The turn, crash it must not.
            return []
        try:
            query = "query: " + _query_from_question(question)   # the e5 query prefix, mandatory it is.
            vec = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores, idxs = self._index.search(vec, self.top_k)
            docs: list[RetrievedDoc] = []
            for score, i in zip(scores[0], idxs[0]):
                if i < 0 or i >= len(self._docs):
                    continue
                if float(score) < self.min_score:   # Off-topic match -- below the cosine floor, skip it.
                    continue
                d = self._docs[i]
                docs.append(RetrievedDoc(
                    doc_id=str(d.get("doc_id", i)),
                    text=str(d.get("text", ""))[: self.char_limit],
                    source=str(d.get("source", "corpus")),
                    score=float(score),
                ))
            return docs
        except Exception:
            return []


# --------------------------------------------------------------------------- #
# The facade -- per-question routing between the backends.
# --------------------------------------------------------------------------- #

class Retriever:
    """The one retriever the pipeline holds -- per QUESTION, the right backend it picks.

    News question?  -> the live web (`WebSearchRetriever`), with Wikipedia as a safety net.
    Anything else?  -> dense FAISS over the local corpus, OR live Wikipedia when no index there is.

    `source` the strategy chooses (from `RetrievalConfig.source`):
      "routed" (default) -- News->web(+wiki fallback), else->faiss-or-wikipedia.  <- "both", the user picked.
      "wikipedia"        -- always live Wikipedia (the existing live.yaml default).
      "web"              -- always DuckDuckGo web search.
      "faiss"            -- always the local corpus (an `index_path` it needs).

    Backends LAZILY constructed they are (and FAISS even more lazily loads its model) -- so
    `Retriever(...)`, cheap and side-effect-free it stays until the first `retrieve`.
    Signature back-compatible with the notebook's `Retriever(top_k=...)` call, it remains.
    """

    def __init__(
        self,
        top_k: int = 3,
        source: str = "routed",
        index_path: Optional[str] = None,
        bm25_index_path: Optional[str] = None,
        embedder: str = "intfloat/multilingual-e5-small",
        char_limit: int = 600,
        timeout_s: float = 6.0,
        min_score: float = 0.0,
        news_fetch_bodies: int = 0,
        news_body_mode: str = "ddg",
        guardian_api_key: str = "",
    ):
        self.top_k = top_k
        self.source = (source or "routed").lower()
        self.index_path = index_path
        self.bm25_index_path = bm25_index_path   # local enwiki BM25; knowledge questions hit it FIRST.
        self.embedder = embedder
        self.char_limit = char_limit
        self.timeout_s = timeout_s
        self.min_score = min_score   # The FAISS cosine floor, to the corpus backend passed it is.
        self.news_fetch_bodies = max(0, int(news_fetch_bodies))   # News web: how many article bodies to pull.
        self.news_body_mode = (news_body_mode or "ddg").lower()   # ... and how: off | ddg | browser.
        self.guardian_api_key = guardian_api_key   # News: the FAST primary body source (when set).
        # The backends, on first use built they are -- a dict of name -> instance, cached here.
        self._cache: dict[str, object] = {}

    # -- backend builders (memoised) --

    def _wikipedia(self) -> WikipediaRetriever:
        if "wikipedia" not in self._cache:
            self._cache["wikipedia"] = WikipediaRetriever(
                top_k=self.top_k, chars_per_doc=self.char_limit, timeout=self.timeout_s,
            )
        return self._cache["wikipedia"]  # type: ignore[return-value]

    def _web(self) -> WebSearchRetriever:
        if "web" not in self._cache:
            # Headlines only -> 400 chars plenty. Bodies fetched -> a larger per-doc budget we leave
            # (1200): option-aware focusing keeps the lead + the windows around the answer terms, so the
            # mid-article attribution sentence fits alongside the setup.
            web_chars = 1200 if self.news_fetch_bodies > 0 else min(self.char_limit, 400)
            self._cache["web"] = WebSearchRetriever(
                top_k=self.top_k, char_limit=web_chars, timeout_s=self.timeout_s,
                fetch_bodies=self.news_fetch_bodies, body_mode=self.news_body_mode,
                guardian_api_key=self.guardian_api_key,
            )
        return self._cache["web"]  # type: ignore[return-value]

    def _faiss(self) -> Optional[FaissRetriever]:
        # No index path -- a FAISS backend, build it we cannot. None we return, and the caller falls back.
        if not self.index_path:
            return None
        if "faiss" not in self._cache:
            self._cache["faiss"] = FaissRetriever(
                index_path=self.index_path, embedder=self.embedder,
                top_k=self.top_k, char_limit=self.char_limit, min_score=self.min_score,
            )
        return self._cache["faiss"]  # type: ignore[return-value]

    def _bm25(self):
        """The local enwiki BM25 backend -- None when no `bm25_index_path`, so the caller falls back.

        Construction is LAZY and crash-safe: a missing/half-built index or a missing `bm25s` dep yields
        None here (logged once), and the knowledge route drops to FAISS/live Wikipedia -- a live turn the
        absent corpus never sinks.
        """
        if not self.bm25_index_path:
            return None
        if "bm25" not in self._cache:
            try:
                from retrieval.bm25_retriever import BM25Retriever
                self._cache["bm25"] = BM25Retriever(
                    index_dir=self.bm25_index_path, top_k=self.top_k, char_limit=self.char_limit,
                )
            except Exception:
                self._cache["bm25"] = None  # build failed -> remember the miss, fall back every time.
        return self._cache["bm25"]

    # -- the public route --

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        """Per the source strategy, the right backend dispatch -- RAW docs out, always."""
        if self.source == "wikipedia":
            return self._wikipedia().retrieve(question)
        if self.source == "web":
            return self._web().retrieve(question)
        if self.source == "bm25":
            bm25_be = self._bm25()
            return bm25_be.retrieve(question) if bm25_be else []
        if self.source == "faiss":
            faiss_be = self._faiss()
            return faiss_be.retrieve(question) if faiss_be else []

        # "routed" / "both" / "hybrid" -- the default: by the question, decide we do.
        if _looks_like_news(question):
            # NEWS -- the live web first, for the post-cutoff facts Wikipedia cannot hold.
            docs = self._web().retrieve(question)
            if docs:
                return docs
            # The web blocked us (a 429, a layout shift) -- Wikipedia, a best-effort net it casts.
            return self._wikipedia().retrieve(question)

        # KNOWLEDGE -- LOCAL FIRST, live only on a miss. Order chosen to dodge the live-API 429s:
        #   1. local enwiki BM25 (offline, unlimited) -- answers the vast majority of evergreen trivia;
        #   2. local FAISS corpus (if a dense index is configured instead/as-well);
        #   3. live Wikipedia -- the safety net for the FEW post-cutoff / niche questions the dump lacks.
        # Each tier returns [] on a miss and we drop to the next, so live fires on <5% of questions.
        bm25_be = self._bm25()
        if bm25_be is not None:
            docs = bm25_be.retrieve(question)
            if docs:
                return docs
        faiss_be = self._faiss()
        if faiss_be is not None:
            docs = faiss_be.retrieve(question)
            if docs:
                return docs
        return self._wikipedia().retrieve(question)


def build_retriever(retrieval_cfg, **overrides) -> Optional[Retriever]:
    """A `RetrievalConfig` -> a wired `Retriever` (or None when disabled). The factory, this is.

    From the config the strategy and knobs it reads; `**overrides` win, for a notebook ablation handy.
    `enabled=False` -> None, so the pipeline its retrieval stage skips entirely.
    """
    if not getattr(retrieval_cfg, "enabled", False):
        return None
    return Retriever(
        top_k=overrides.get("top_k", getattr(retrieval_cfg, "top_k", 3)),
        source=overrides.get("source", getattr(retrieval_cfg, "source", "routed")),
        index_path=overrides.get("index_path", getattr(retrieval_cfg, "index_path", None)),
        bm25_index_path=overrides.get("bm25_index_path", getattr(retrieval_cfg, "bm25_index_path", None)),
        embedder=overrides.get("embedder", getattr(retrieval_cfg, "embedder", "intfloat/multilingual-e5-small")),
        min_score=overrides.get("min_score", getattr(retrieval_cfg, "min_score", 0.0)),
        news_fetch_bodies=overrides.get(
            "news_fetch_bodies", getattr(retrieval_cfg, "news_fetch_bodies", 0)
        ),
        news_body_mode=overrides.get(
            "news_body_mode", getattr(retrieval_cfg, "news_body_mode", "ddg")
        ),
        guardian_api_key=overrides.get(
            "guardian_api_key", getattr(retrieval_cfg, "guardian_api_key", "")
        ),
    )
