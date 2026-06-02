"""BM25 retrieval over the local enwiki dump -- the offline, rate-limit-free knowledge backend.

`retrieve(question) -> [RetrievedDoc]`, the same shape every backend speaks. The question text is the
BM25 query (we do NOT AND the options in -- that pulls all four answers' topics in as noise, the News
lesson); BM25's IDF down-weights the scaffolding words on its own. Each matched ARTICLE is then trimmed
to the answer with the shared option-term FOCUS (retrieval._focus), so the model gets a tight excerpt --
the lead plus the windows where the option terms appear -- not a 50KB article.

The `BM25Index` is heavy to construct, so it is LAZY: built on the first retrieve (or injected, for tests).
Crash-safe ALWAYS -- a missing index / load error / query slip returns [] and the caller falls back to the
live Wikipedia net (so a half-built index never sinks a live turn).
"""
from __future__ import annotations

from retrieval._focus import focus as _focus_text
from retrieval._focus import option_terms as _option_terms_of
from schemas import Question, RetrievedDoc


class BM25Retriever:
    """A question -> top-k FOCUSED local-corpus excerpts (BM25 ranked). Lazy index, crash-safe."""

    def __init__(
        self,
        index_dir: str | None = None,
        backend=None,                 # an object with .search(query, k) -- inject for tests; else BM25Index.
        top_k: int = 3,
        char_limit: int = 500,        # the lead (article head) kept as the topic anchor.
        chars_focus: int = 1100,      # total kept per doc (lead + answer-term windows).
        focus_window: int = 180,      # chars either side of an option-term hit.
        query_chars: int = 300,       # the question, capped to this as the BM25 query.
    ):
        self.index_dir = index_dir
        self._backend = backend       # None until first use (or test-injected).
        self.top_k = top_k
        self.char_limit = char_limit
        self.chars_focus = chars_focus
        self.focus_window = focus_window
        self.query_chars = query_chars

    def _get_backend(self):
        """The BM25Index, built lazily on first use -- a load failure raises (caught by `retrieve`)."""
        if self._backend is None:
            from retrieval.bm25_index import BM25Index  # imported late: bm25s is an optional dep.
            self._backend = BM25Index(self.index_dir)
        return self._backend

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        """Top-k BM25 articles, each focused to the answer window. [] on ANY failure (live net catches)."""
        try:
            query = (question.text or "").strip()[: self.query_chars]
            if not query:
                return []
            hits = self._get_backend().search(query, k=self.top_k)
            if not hits:
                return []
            terms = _option_terms_of(question)
            docs: list[RetrievedDoc] = []
            for doc_id, text, source, score in hits:
                focused = _focus_text(
                    text, terms, lead=text,
                    chars_focus=self.chars_focus, focus_window=self.focus_window,
                    chars_lead=self.char_limit,
                )
                docs.append(RetrievedDoc(
                    doc_id=str(doc_id),
                    text=focused,
                    source=str(source),
                    score=float(score),
                ))
            return docs
        except Exception:
            return []  # missing/half-built index, query slip -> no evidence; the caller falls back to live.
