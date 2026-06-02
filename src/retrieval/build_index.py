"""Build a FAISS index for `FaissRetriever`, this script does -- on Colab, once per corpus run it.

A corpus JSONL in, an index directory out:
    <out>/index.faiss   -- the dense vectors (inner-product over normalised e5 embeddings => cosine).
    <out>/docs.jsonl    -- one {doc_id, text, source} per line, ROW-ALIGNED to the FAISS vectors.
`FaissRetriever` those two files reads; row i of docs.jsonl, vector i in the index it is.

The corpus a JSONL is, one record per line. Flexible the field names are (`--text-field` etc.); the
text we MUST have, an id/source nice-to-have they are. Long records into ~`--chunk-chars` chunks we split
(a focused chunk, the retriever it helps; a whole article, the e5 context it overflows).

e5 a prefix demands: PASSAGES "passage: " here at index time, QUERIES "query: " in the retriever.
Mismatch them, and the cosine scores garbage they are. So the prefix, baked in here it is.

A ready free corpus, the course suggests: the **Simple Wikipedia** dump
`simplewiki-2020-11-01.jsonl.gz` (each line a `{title, text}` page). Decompress, then:
    python -m retrieval.build_index --corpus simplewiki.jsonl --out data/corpus/simplewiki \
        --text-field text --id-field title --source-field title

Run it from `src/` (so `retrieval` importable is), or with `src` on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator


def _iter_records(path: Path, max_docs: int | None) -> Iterator[dict]:
    """The corpus JSONL, line by line stream it we do -- a huge dump into memory, load it we will not."""
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # A malformed line, skip it we do -- the build, sink it must not.
            n += 1
            if max_docs is not None and n >= max_docs:
                return


def _chunk(text: str, chunk_chars: int) -> list[str]:
    """A long text into ~chunk_chars pieces, split it we do -- on paragraph/sentence joints, gently."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= chunk_chars:
        return [text] if text else []
    chunks: list[str] = []
    # Sentence-ish boundaries, prefer we do -- a chunk mid-word, cut it we would rather not.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= chunk_chars:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                chunks.append(buf)
            # A single sentence longer than the budget -- hard-split it we must.
            while len(s) > chunk_chars:
                chunks.append(s[:chunk_chars])
                s = s[chunk_chars:]
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def build(
    corpus: str,
    out: str,
    embedder: str = "intfloat/multilingual-e5-small",
    text_field: str = "text",
    id_field: str | None = None,
    source_field: str | None = None,
    chunk_chars: int = 500,
    batch_size: int = 256,
    max_docs: int | None = None,
) -> None:
    # Heavy deps, only when actually building -- importing this module, a GPU it must never wake.
    import faiss  # type: ignore
    import numpy as np  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    corpus_path = Path(corpus)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading corpus: {corpus_path}")
    texts: list[str] = []     # The "passage: "-prefixed strings to embed, these are.
    docs: list[dict] = []     # The row-aligned {doc_id, text, source} metadata, this is.

    for rec in _iter_records(corpus_path, max_docs):
        raw = rec.get(text_field)
        if not raw:
            continue
        base_id = str(rec.get(id_field)) if id_field and rec.get(id_field) is not None else None
        source = str(rec.get(source_field)) if source_field and rec.get(source_field) is not None else "corpus"
        for j, chunk in enumerate(_chunk(str(raw), chunk_chars)):
            doc_id = f"{base_id}#{j}" if base_id else f"doc{len(docs)}"
            docs.append({"doc_id": doc_id, "text": chunk, "source": source})
            texts.append(f"passage: {chunk}")   # the e5 passage prefix, mandatory it is.

    if not texts:
        raise SystemExit("No usable records found -- the --text-field correct is? Empty the corpus is?")

    print(f"Encoding {len(texts)} chunks with {embedder} ...")
    model = SentenceTransformer(embedder)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,    # normalised => inner product IS cosine similarity.
        show_progress_bar=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)    # exact cosine search; for a coursework corpus, fast enough it is.
    index.add(embeddings)

    idx_file = out_dir / "index.faiss"
    docs_file = out_dir / "docs.jsonl"
    faiss.write_index(index, str(idx_file))
    with docs_file.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Done. {len(docs)} chunks, dim={dim}.")
    print(f"  {idx_file}")
    print(f"  {docs_file}")
    print(f"Point RetrievalConfig.index_path at: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a FAISS index for FaissRetriever.")
    p.add_argument("--corpus", required=True, help="Path to the corpus JSONL.")
    p.add_argument("--out", required=True, help="Output index directory (gets index.faiss + docs.jsonl).")
    p.add_argument("--embedder", default="intfloat/multilingual-e5-small")
    p.add_argument("--text-field", default="text")
    p.add_argument("--id-field", default=None)
    p.add_argument("--source-field", default=None)
    p.add_argument("--chunk-chars", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-docs", type=int, default=None, help="Cap records read (for a quick smoke build).")
    args = p.parse_args()
    build(
        corpus=args.corpus, out=args.out, embedder=args.embedder,
        text_field=args.text_field, id_field=args.id_field, source_field=args.source_field,
        chunk_chars=args.chunk_chars, batch_size=args.batch_size, max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
