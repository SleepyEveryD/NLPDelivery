"""Flatten a WikiExtractor dump into the `articles.jsonl` that `bm25_index.build_bm25` consumes.

The pipeline (run OFFLINE, once -- it is heavy):

  1. Download the full-enwiki dump (~22 GB):
       wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2

  2. Extract clean plaintext with WikiExtractor (strips wikitext/templates -> one JSON per article):
       pip install wikiextractor
       python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles-multistream.xml.bz2 \
           --json --no-templates --processes 4 -o enwiki_extracted

  3. THIS script -- flatten WikiExtractor's tree into one articles.jsonl ({title, text, source}):
       python -m retrieval.prepare_enwiki --input enwiki_extracted \
           --out data/corpus/enwiki/articles.jsonl

  4. Build the BM25 index:
       python -m retrieval.bm25_index --corpus data/corpus/enwiki/articles.jsonl \
           --out data/corpus/enwiki/bm25_index

  5. Point the config at it (configs/live.yaml):  retrieval.bm25_index_path: data/corpus/enwiki/bm25_index

WikiExtractor emits a directory tree (AA/wiki_00, AA/wiki_01, ... AB/...), each LINE a JSON object
{id, revid, url, title, text}. We keep RAW `text` (already plaintext), drop redirects/stubs/disambiguation,
and write one compact record per line. RAW content only (D-008) -- we never generate, only reshape.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator


def _iter_extracted(input_path: Path) -> Iterator[dict]:
    """Every article JSON under a WikiExtractor output dir (or a single file), streamed line by line."""
    files = sorted(input_path.rglob("wiki_*")) if input_path.is_dir() else [input_path]
    for fp in files:
        try:
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a malformed line, skip -- the flatten must not sink.
        except OSError:
            continue


def _is_keepable(title: str, text: str, min_chars: int) -> bool:
    """Drop redirects (empty/near-empty body), tiny stubs, and disambiguation pages -- noise, these are."""
    if not title or not text:
        return False
    if len(text) < min_chars:
        return False
    low = title.lower()
    if low.startswith(("list of", "index of")):
        return False
    if "(disambiguation)" in low:
        return False
    if text.lstrip().lower().startswith("redirect"):
        return False
    return True


def flatten(input_dir: str, out_path: str, min_chars: int = 200, max_docs: int | None = None) -> int:
    """WikiExtractor tree -> articles.jsonl ({title, text, source}). Returns the count written."""
    inp = Path(input_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as w:
        for rec in _iter_extracted(inp):
            title = (rec.get("title") or "").strip()
            text = (rec.get("text") or "").strip()
            if not _is_keepable(title, text, min_chars):
                continue
            source = rec.get("url") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            w.write(json.dumps({"title": title, "text": text, "source": source}, ensure_ascii=False) + "\n")
            n += 1
            if max_docs is not None and n >= max_docs:
                break
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten a WikiExtractor dump into articles.jsonl.")
    ap.add_argument("--input", required=True, help="WikiExtractor output dir (or a single wiki_* file).")
    ap.add_argument("--out", required=True, help="Output articles.jsonl path.")
    ap.add_argument("--min-chars", type=int, default=200, help="Drop articles shorter than this (stubs).")
    ap.add_argument("--max-docs", type=int, default=None, help="Cap (for a quick smoke build).")
    args = ap.parse_args()
    n = flatten(args.input, args.out, min_chars=args.min_chars, max_docs=args.max_docs)
    print(f"wrote {n} articles -> {args.out}")


if __name__ == "__main__":
    main()
