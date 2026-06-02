"""Configuration loading. By one config object, the whole run is described.

YAML in, a typed object out -- and reproducible an experiment becomes, when its config is logged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    quantization: str = "4bit"            # bitsandbytes nf4 -- the Colab-friendly default, this is.
    max_new_tokens: int = 256
    temperature: float = 0.0              # Greedy by default -- deterministic and fast, we stay.
    dtype: str = "bfloat16"


@dataclass
class RetrievalConfig:
    enabled: bool = False                 # Off for the baseline -- on later it comes.
    source: str = "wikipedia"             # "wikipedia" (live API) | "faiss" (local corpus). Phase 4.
    top_k: int = 3
    embedder: str = "intfloat/multilingual-e5-small"
    index_path: Optional[str] = None
    # A LOCAL BM25 index over the full-enwiki dump (built with retrieval.bm25_index). When set, knowledge
    # questions hit it FIRST -- offline, millisecond, NO rate limit -- and only fall through to live
    # Wikipedia when it returns nothing (post-cutoff / niche). This is the cure for the live-API 429s that
    # killed ~half the Entertainment turns. None -> BM25 skipped, the old FAISS-or-Wikipedia path stands.
    # Build it once (see retrieval.prepare_enwiki + retrieval.bm25_index) and point this at the index dir.
    bm25_index_path: Optional[str] = None
    # Minimum FAISS cosine similarity to KEEP a corpus doc. 0.0 = off (every top_k doc passes, the
    # old behaviour). A floor (~0.72 for e5-small) drops off-topic matches -- the second line of
    # defence after the needs_retrieval gate, for when retrieval fires but the corpus has no real hit
    # (e.g. a name-collision pulling celebrity pages). Below the floor everything? -> [] and the
    # routed retriever falls back to live Wikipedia. Web/Wikipedia docs (no real score) are unaffected.
    min_score: float = 0.0
    # NEWS only: how many top Google-News-RSS articles to fetch the BODY of (0 = headlines only). The
    # headline carries the gist, but "who was quoted.." / exact numbers live in the article TEXT (qid
    # 11415 died with headlines alone). Best-effort + crash-safe: a fetch fail -> that item keeps just
    # its headline. Each body is ONE extra HTTP with a TIGHT timeout, so the 30s wall it respects.
    news_fetch_bodies: int = 0
    # HOW to fetch those bodies: "off" | "ddg" | "browser".
    #   "ddg"     -- DuckDuckGo gives the DIRECT publisher URL, `requests` fetches it. Fast, but DDG is
    #                BLOCKED on the Colab IP (returns nothing there) -- so on Colab this yields no bodies.
    #   "browser" -- a headless Chromium opens the Google-News link, RUNS the JS (past the consent wall +
    #                redirect) and reads the rendered article. The ONLY path that gets bodies on Colab; it
    #                needs `playwright install chromium` and is ~3-4s/article. Name "headless Chromium" in the video.
    news_body_mode: str = "ddg"
    # The Guardian Open Platform key. When set, the News body comes FIRST from the Guardian Content API
    # (free, raw `bodyText` in ONE ~0.2s call -- no browser, no consent wall) and only NON-Guardian stories
    # fall back to `news_body_mode`. A SECRET it is -- NEVER in this YAML; the notebook injects it from a
    # Colab secret. Empty -> the Guardian path simply skipped. Name "Guardian Open Platform API" in the video.
    guardian_api_key: str = ""


@dataclass
class GameConfig:
    """Live-game ("real test") settings -- ignored when mode is offline, they are."""
    competition_id: int = 0               # Which competition (0..5); the topic it picks.
    game_mode: str = "text"               # "text" | "speech" -- how the question reaches us.
    aim_seconds: float = 25.0             # Answer-by target; the network margin below the 30s wall, this leaves.


@dataclass
class RunConfig:
    run_id: str = "dev"
    seed: int = 13
    latency_budget_s: float = 30.0        # The hard wall of the game, this is.
    prompt_strategy: str = "zero_shot_v1"
    # The run mode: "offline" (our own dev-set test) | "live" (the real game API). See schemas.RunMode.
    mode: str = "offline"
    # Where the offline dev set lives -- read only when mode is offline, it is.
    dataset_path: str = "data/dev_questions.jsonl"
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    game: GameConfig = field(default_factory=GameConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        # Read the YAML, into a typed config turn it we do.
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        model = ModelConfig(**(data.pop("model", {}) or {}))
        retrieval = RetrievalConfig(**(data.pop("retrieval", {}) or {}))
        game = GameConfig(**(data.pop("game", {}) or {}))
        return cls(model=model, retrieval=retrieval, game=game, **data)

    def to_dict(self) -> dict[str, Any]:
        # For the run's meta.json, a plain dict this gives.
        return asdict(self)
