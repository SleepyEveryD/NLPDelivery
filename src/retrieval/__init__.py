"""retrieval package. A module of the PoliMillionaire system, this is.

All backends the RAW-evidence rule (D-008) honour -- raw chunks only, never a generated answer:
- `WikipediaRetriever` -- the live, free Wikipedia API (knowledge topics; no index to build).
- `WebSearchRetriever` -- the live, keyless DuckDuckGo search (post-cutoff NEWS).
- `FaissRetriever`     -- a local FAISS corpus (the course's dense RAG; build with `build_index.py`).
- `Retriever`          -- the facade the pipeline holds; per QUESTION it routes News->web, else->dense/wiki.
- `build_retriever`    -- a `RetrievalConfig` -> a wired `Retriever` (or None when disabled).
"""
from .wikipedia import WikipediaRetriever
from .bm25_retriever import BM25Retriever
from .retriever import (
    FaissRetriever,
    Retriever,
    WebSearchRetriever,
    build_retriever,
)

__all__ = [
    "WikipediaRetriever",
    "BM25Retriever",
    "FaissRetriever",
    "Retriever",
    "WebSearchRetriever",
    "build_retriever",
]
