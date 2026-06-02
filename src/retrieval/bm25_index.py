"""A BM25 index over a LOCAL corpus (the full-enwiki dump) -- build it once, query it offline.

WHY local BM25, and not the live Wikipedia API: forcing retrieval on every knowledge question hammered
the live API and ~half the turns died to HTTP 429 (rate limit) on Colab's shared IP. A local index is the
cure -- unlimited, millisecond, no network. BM25 (lexical) is the right engine here: entity/keyword-heavy
trivia ("Shawshank Redemption filming location", "Whitney Houston debut album") is exactly where BM25
shines, and -- unlike a dense index -- it needs NO GPU and NO multi-hour embedding pass over 7M articles.

Engine: `bm25s` (a fast, scipy-sparse BM25). The index SAVES to disk and LOADS memory-mapped, so query-time
RAM stays small even for full enwiki; the heavy step is the one-time BUILD (do it on a >=32GB-RAM machine,
or on a curated subset -- see prepare_enwiki.py). We index at ARTICLE granularity (one doc per article);
the per-question option-term FOCUS (retrieval._focus) trims the matched article to the answer window at
query time, so the index stays ~7M docs (not ~30M chunks) and the model still gets a tight excerpt.

Layout of an index dir:
    <dir>/bm25/        -- the bm25s index (its own save format).
    <dir>/offsets.npy  -- int64[n_docs]; byte offset of each doc's line in the corpus jsonl (row-aligned
                          to the bm25s doc order, so doc index i -> line at offsets[i]).
    <dir>/meta.json    -- {corpus_path, n_docs, text_field, id_field, source_field}.

RAW content only (D-008): we index and return RAW article text, never a generated answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def _require_bm25s():
    """Import bm25s, or raise a friendly install hint -- the heavy dep we keep optional & lazy."""
    try:
        import bm25s  # noqa: F401
        return bm25s
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "bm25s is required for the local BM25 index. Install it: `pip install bm25s`. "
            "(Loading/querying is light; the one-time build wants >=32GB RAM for full enwiki.)"
        ) from exc


# A cheap, dependency-free tokenizer for QUERIES (the build uses bm25s.tokenize for the corpus; we mirror
# its lowercase+split here so query terms line up). Stemming we skip -- it needs PyStemmer; plain works.
_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOK_RE.findall((text or "").lower())


def build_bm25(
    corpus_path: str,
    out_dir: str,
    text_field: str = "text",
    id_field: str = "title",
    source_field: str = "title",
    max_docs: int | None = None,
) -> str:
    """A corpus JSONL (one article per line) -> a saved BM25 index dir. Returns the out_dir.

    Streamed in line order; the bm25s doc order MATCHES the file's line order, and we record each line's
    byte offset so the retriever can seek the raw article back by doc index. The tokenized corpus is held
    in RAM for the index build (the one heavy step) -- for full enwiki, a high-RAM machine this wants.
    """
    bm25s = _require_bm25s()
    corpus_p = Path(corpus_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    offsets: list[int] = []
    n = 0
    # Manual readline so we capture the byte offset of EACH line (f.tell() before the read).
    with corpus_p.open("rb") as fb:
        while True:
            pos = fb.tell()
            raw = fb.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = (rec.get(text_field) or "").strip()
            if not txt:
                continue
            offsets.append(pos)
            texts.append(txt)
            n += 1
            if max_docs is not None and n >= max_docs:
                break

    if not texts:
        raise ValueError(f"No usable records in {corpus_path} (text_field={text_field!r}).")

    # Tokenize + index (bm25s owns the heavy sparse build). Stopwords pruned; no stemmer (no extra dep).
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=True)
    del texts  # free the raw text RAM before the index allocates its sparse matrices.
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(str(out / "bm25"))

    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    (out / "meta.json").write_text(json.dumps({
        "corpus_path": str(corpus_p.resolve()),
        "n_docs": n,
        "text_field": text_field,
        "id_field": id_field,
        "source_field": source_field,
    }), encoding="utf-8")
    return str(out)


class BM25Index:
    """A built BM25 index, loaded for QUERYING -- `search(query, k) -> [(doc_id, text, source, score)]`.

    The bm25s index loads memory-mapped (small RAM); the raw article text we seek lazily from the corpus
    jsonl via the byte offsets, so only the top-k articles are ever read off disk per query.
    """

    def __init__(self, index_dir: str, mmap: bool = True):
        bm25s = _require_bm25s()
        self.dir = Path(index_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.offsets = np.load(self.dir / "offsets.npy")
        self._retriever = bm25s.BM25.load(str(self.dir / "bm25"), mmap=mmap)
        self._corpus_path = Path(self.meta["corpus_path"])
        self._text_field = self.meta.get("text_field", "text")
        self._id_field = self.meta.get("id_field", "title")
        self._source_field = self.meta.get("source_field", "title")
        self._fh = self._corpus_path.open("rb")  # kept open for seeks.

    def _record_at(self, doc_idx: int) -> dict | None:
        """The raw corpus record at bm25s doc index `doc_idx` -- seek by byte offset, one line read."""
        if doc_idx < 0 or doc_idx >= len(self.offsets):
            return None
        self._fh.seek(int(self.offsets[doc_idx]))
        line = self._fh.readline().decode("utf-8", "ignore").strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def search(self, query: str, k: int = 3) -> list[tuple[str, str, str, float]]:
        """Top-k articles for `query` -> [(doc_id, raw_text, source, score)]. Empty list on miss/failure."""
        bm25s = _require_bm25s()
        toks = bm25s.tokenize([query or ""], stopwords="en", show_progress=False)
        k = max(1, min(k, len(self.offsets)))
        results, scores = self._retriever.retrieve(toks, k=k, show_progress=False)
        out: list[tuple[str, str, str, float]] = []
        for doc_idx, score in zip(results[0].tolist(), scores[0].tolist()):
            rec = self._record_at(int(doc_idx))
            if not rec:
                continue
            text = (rec.get(self._text_field) or "").strip()
            if not text:
                continue
            doc_id = str(rec.get(self._id_field) or doc_idx)
            source = str(rec.get(self._source_field) or doc_id)
            out.append((doc_id, text, source, float(score)))
        return out


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build a local BM25 index over an articles.jsonl corpus.")
    ap.add_argument("--corpus", required=True, help="articles.jsonl (one article per line).")
    ap.add_argument("--out", required=True, help="Output index directory.")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--id-field", default="title")
    ap.add_argument("--source-field", default="source")
    ap.add_argument("--max-docs", type=int, default=None, help="Cap (for a quick smoke build).")
    args = ap.parse_args()
    out = build_bm25(
        args.corpus, args.out,
        text_field=args.text_field, id_field=args.id_field, source_field=args.source_field,
        max_docs=args.max_docs,
    )
    print(f"BM25 index built -> {out}")


if __name__ == "__main__":
    _main()
