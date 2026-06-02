"""Adapter over the course-provided `millionaire_client` package.

The real game client, the course gives us (millionaire_client). Reimplement it we do NOT --
wrap it we do, and its Question/Option into our schemas.Question we translate.

Setup: sign up first in a browser at http://131.175.15.22:51111/ (a PoliMi email, one account).
On sys.path the `millionaire_client` package must be (on Colab, kept in Drive it is).
Polite we stay: rapid consecutive games/requests, avoid them we do.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from schemas import Question, QuestionType

# The option letters, in order these are (their server gives integer ids, not letters).
_LETTERS = "ABCDEFGH"

# A single "A) text" / "B. text" / "C: text" option line, this matches (the leading letter + its text).
_OPT_LINE = re.compile(r"^\s*([A-Za-z])\s*[).:\-]\s*(.+?)\s*$")


def _norm(s: str) -> str:
    # For comparison only: lowercased, whitespace collapsed, a trailing period dropped.
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def strip_embedded_options(text: str, option_texts: list[str]) -> str:
    """A trailing 'A) ... B) ...' block, baked into the question text it sometimes is -- remove it we do.

    The danger (qid 617): the server's option ORDER differs from the text's embedded letters, so two
    conflicting A/B/C/D schemes the model would see. Strip the block we do -- but ONLY when its texts
    truly duplicate the server options (a subset match) AND the letters run A, B, C, ... in order.
    A question that merely mentions a letter, never mangle we will. Lossless this is -- the options,
    in EvalRecord.options preserved they already are.
    """
    if not text or not option_texts:
        return text
    lines = text.splitlines()
    while lines and not lines[-1].strip():  # Trailing blank lines, first drop them we do.
        lines.pop()

    letters: list[str] = []
    texts: list[str] = []
    n = len(lines)
    while n > 0:  # From the bottom up, the contiguous run of option-shaped lines we gather.
        m = _OPT_LINE.match(lines[n - 1])
        if not m:
            break
        letters.append(m.group(1).upper())
        texts.append(m.group(2))
        n -= 1
    letters.reverse()
    texts.reverse()

    if len(texts) < 2:  # Too short to be an options block, it is.
        return text
    # Consecutive A, B, C, ... the letters must be -- else a real list this may be, untouched leave it.
    if letters != [chr(ord("A") + i) for i in range(len(letters))]:
        return text
    # And a subset of the server options the texts must be -- only a true duplicate, strip we do.
    server = {_norm(t) for t in option_texts}
    if not {_norm(t) for t in texts} <= server:
        return text

    cleaned = "\n".join(lines[:n]).rstrip()
    return cleaned or text  # All text was the block? Then keep the original, defensive we stay.


def adapt_question(api_q, topic: str | None = None) -> Question:
    """Their Question (options carry integer ids) -> our schemas.Question (with a letter->id map).

    The letter->id map, keep it we must -- by integer id the server wants the answer, not by letter.
    `topic`: the competition's name, the live loop passes it -- the RELIABLE topic signal (the per-turn
    `level` aside, the server tells us nothing per question). Else None it stays, and the classifier the
    topic must GUESS from text (`_infer_topic`) -- which under-fired retrieval on clear history questions.
    """
    options: dict[str, str] = {}
    option_ids: dict[str, int] = {}
    for i, opt in enumerate(api_q.options):
        letter = _LETTERS[i]
        options[letter] = opt.text
        option_ids[letter] = opt.id
    # A duplicated options block baked into the text, strip it we do -- the double-lettering, gone (qid 617).
    text = strip_embedded_options(api_q.text or "", list(options.values()))
    return Question(
        qid=str(api_q.id),
        text=text,
        options=options,
        option_ids=option_ids,
        qtype=QuestionType.MCQ,
        level=getattr(api_q, "level", None),
        topic=topic,
    )


class GameClient:
    """A thin wrapper over MillionaireClient. Login, list competitions, and play with OUR pipeline, it lets us.

    The provided package does the HTTP; the translation to/from our schemas, this class owns.
    """

    def __init__(self, base_url: str = "http://131.175.15.22:51111/", timeout: int = 30):
        try:
            from millionaire_client import MillionaireClient
        except ImportError as e:  # On sys.path the provided package must be, else fail loudly we do.
            raise ImportError(
                "The provided 'millionaire_client' package on sys.path it must be "
                "(folder NLP_assignment_api_client/). See README + PoliMillionaire.ipynb, you should."
            ) from e
        self._client = MillionaireClient(base_url, timeout=timeout)
        # The course server, idle keep-alive sockets it drops -- while our LOCAL model thinks for
        # seconds, stale the connection goes, and the next request a RemoteDisconnected raises. A
        # retrying adapter, mount we do: transient drops + 5xx it reconnects-and-retries, so mid-game
        # the run never sinks (a single dropped connection, a whole competition it must not cost).
        self._harden_session()

    def _harden_session(self) -> None:
        """Against flaky-server drops, the underlying requests.Session armour we give.

        A urllib3 Retry, on connection errors AND 5xx it fires, with capped exponential back-off.
        On a dropped keep-alive socket, a FRESH connection the retry opens -- the stale-socket curse,
        broken it is. Best-effort entirely: the package internals shift / urllib3 differs -> silently
        skip we do (unhardened but alive, the client stays).
        """
        try:
            from requests.adapters import HTTPAdapter
            try:
                from urllib3.util.retry import Retry
            except ImportError:  # A much older urllib3, a different import path it has.
                from urllib3.util import Retry  # type: ignore

            # ALL methods retry (POST included) -- "without response" means the server saw nothing,
            # so safe a re-send is. The 30s wall, the capped back-off respects (0.6, 1.2, 2.4s ...).
            retry_kwargs = dict(
                total=4, connect=4, read=2, backoff_factor=0.6,
                status_forcelist=(500, 502, 503, 504), raise_on_status=False,
            )
            try:
                retry = Retry(allowed_methods=None, **retry_kwargs)
            except TypeError:  # urllib3 < 1.26, the old keyword name it wants.
                retry = Retry(method_whitelist=None, **retry_kwargs)

            adapter = HTTPAdapter(max_retries=retry)
            session = self._client._base._session  # The provided package's requests.Session, this is.
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Keep-alive OFF, the root cure it is. A LONG gap between turns -- retrieval-heavy rounds
            # (Ancient History & Politics: EVERY question fires Wikipedia) the Wikipedia calls stretch --
            # a reused socket lets rot, so on the next answer-submit a RemoteDisconnected we hit. A FRESH
            # connection each request we force ('Connection: close'); the handshake cost, against the 30s
            # wall trivial it is. No stale socket can exist, when none is reused.
            session.headers["Connection"] = "close"
        except Exception:
            pass  # The internals moved -- unhardened, but the game still runs.

    def login(self, username: str, password: str):
        # Authenticate we must, before competitions or games touch we can.
        return self._client.login(username, password)

    def list_competitions(self) -> list:
        # The competitions and their public ids (0,1,2,...), these it returns.
        return self._client.competitions.list_all()

    def my_reached_levels(self, username: str | None = None, mode: str = "text") -> list[dict]:
        """Per competition, MY leaderboard standing fetch -- the server's authoritative reached_level + score.

        The REAL climb, this is. Our local logs it captured not (until now); the SERVER the truth holds,
        so a run already played, re-run we need not -- the leaderboard remembers. `username` omitted ->
        the logged-in user we ask. A competition without an entry for us, skipped it is. Best-effort per
        competition: one lookup fails, the rest it never sinks.

        Returns: a list of dicts {competition_id, name, reached_level, score}, in competition order.
        """
        name = username or self._client.user.username
        out: list[dict] = []
        for comp in self.list_competitions():
            try:
                entry = self._client.leaderboard.find_player(comp.id, name, mode=mode)
            except Exception:
                entry = None  # That board unreachable / no entry -- onward we go, crash we must not.
            if entry is not None:
                out.append({
                    "competition_id": comp.id,
                    "name": comp.name,
                    "reached_level": entry.reached_level,
                    "score": entry.score,
                })
        return out

    def play(
        self,
        competition_id: int,
        answer_fn: Callable[[Question], str],
        on_result: Optional[Callable] = None,
        mode: str = "text",
    ):
        """Play one full game with our pipeline as the strategy.

        Args:
            answer_fn: our Question -> a chosen letter ("A".."D"). The pipeline, this wraps.
            on_result: optional callback (our_question, letter, AnswerResult, time_remaining_s) -- for logging.
            mode: "text" or "speech".

        Note: against the 30s wall the answer_fn must stay -- seed its LatencyGuard from
        `game.time_remaining` minus a network margin, ideally you should.
        """
        game = self._client.game.start(competition_id, mode=mode)
        # The competition name -- the RELIABLE topic, every question of THIS game shares it (D-014).
        # On it the classifier gates retrieval (News / Ancient History -> always retrieve) and the
        # retriever routes (News -> live web). The server sets no per-question topic, so here we inject it.
        comp_topic = getattr(getattr(game.state, "competition", None), "name", None)
        while game.in_progress:
            api_q = game.current_question
            if not api_q:  # No question left -- ended the game has.
                break
            q = adapt_question(api_q, topic=comp_topic)
            time_left = game.time_remaining  # The server's truth on remaining seconds, this is.

            letter = answer_fn(q)  # Here, our pipeline decides.

            option_id = q.option_ids.get((letter or "").strip().upper())
            if option_id is None:
                # Parse failed -- but skip we cannot; guess the first option we must, an answer is owed.
                option_id = next(iter(q.option_ids.values()))

            result = game.answer(option_id)
            if on_result:
                on_result(q, letter, result, time_left)
            if result.game_over or result.timed_out:
                break
        return game
