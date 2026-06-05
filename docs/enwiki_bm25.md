# Local full-enwiki BM25 retrieval

The cure for the live Wikipedia **429 rate-limit** that killed ~half the Entertainment turns: a **local
BM25 index over the full English Wikipedia dump**. Offline, millisecond, unlimited — no shared-IP throttle.

```
enwiki dump (.xml.bz2)
   │  wikiextractor  (clean wikitext → plaintext JSON)
   ▼
articles.jsonl  ({title, text, source})   ← prepare_enwiki.py
   │  bm25s  (tokenize → sparse BM25 → save)
   ▼
bm25_index/  (bm25/, offsets.npy, meta.json)   ← bm25_index.py
   │
   ▼
BM25Retriever  →  option-term FOCUS  →  RetrievedDoc   (wired into Retriever, knowledge-topic LOCAL-FIRST)
```

## Why BM25 (not a dense FAISS rebuild)

- **No GPU, no multi-hour embedding** of ~7M articles. Indexing is a one-time CPU job.
- Entertainment/History trivia is **entity/keyword-heavy** ("Shawshank Redemption filming location") —
  exactly BM25's strength.
- `bm25s` **saves to disk and loads memory-mapped**, so query-time RAM is small even for full enwiki.
- A dense index can still be added later as a *reranker* on top (the FAISS path is untouched).

We index at **article granularity** (one doc per article, ~7M docs — not ~30M chunks), then trim each
matched article to the answer with the shared **option-term focus** (`retrieval/_focus.py`) at query time.

## Build it — one command

```bash
bash scripts/build_enwiki_bm25.sh           # download → extract → flatten → index → smoke query
#   Colab:  !bash scripts/build_enwiki_bm25.sh
#   smoke:  MAX_DOCS=20000 bash scripts/build_enwiki_bm25.sh     # quick partial build to test the chain
```
Each step skips if its output already exists, so a re-run resumes. The manual steps below are what the
script runs, for reference / customisation.

## Build it (offline, once)

> **Resources.** The **build** is the heavy step — for full enwiki give it a machine with **≥32 GB RAM**
> and **~80 GB free disk** (22 GB dump + ~15 GB plaintext + index). **Loading/querying is light** (mmap),
> so a built index runs fine on a normal Colab box. Tight on RAM? Use `--max-docs` for a smoke build, or
> build a curated subset.

```bash
# 1. Download the dump (~22 GB)
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2

# 2. Clean → plaintext JSON
pip install wikiextractor
python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles-multistream.xml.bz2 \
    --json --no-templates --processes 4 -o enwiki_extracted

# 3. Flatten → articles.jsonl   (run from src/, or with src on PYTHONPATH)
python -m retrieval.prepare_enwiki --input enwiki_extracted \
    --out data/corpus/enwiki/articles.jsonl

# 4. Build the BM25 index
pip install bm25s
python -m retrieval.bm25_index --corpus data/corpus/enwiki/articles.jsonl \
    --out data/corpus/enwiki/bm25_index
```

A quick smoke build first (a few thousand articles) is wise:
`python -m retrieval.bm25_index --corpus data/corpus/enwiki/articles.jsonl --out /tmp/bm25_smoke --max-docs 5000`

## Turn it on

In `configs/live.yaml` under `retrieval:`

```yaml
  bm25_index_path: data/corpus/enwiki/bm25_index   # null/omit → BM25 skipped, old live-Wikipedia path stands
```

The index dir is portable — **build it once, host it** (Google Drive / HF Hub) and just point
`bm25_index_path` at the downloaded copy on Colab; no rebuild per session.

## How routing uses it

`Retriever.retrieve` (source `routed`), for a **knowledge** question (Entertainment / History / Science /
Philosophy), tries in order and stops at the first hit:

1. **local enwiki BM25** — answers the vast majority of evergreen trivia, 0 network;
2. **local FAISS** — if a dense `index_path` is also configured;
3. **live Wikipedia** — the safety net for the FEW post-cutoff / niche questions the dump lacks.

So live Wikipedia fires on **<5 %** of questions → the 429s disappear. **News is unchanged** — it keeps its
own live-web path (Google News RSS + Guardian + headless Chromium); BM25 is knowledge-only.

Every backend returns `[]` on a miss/failure and the next tier catches it, so a missing or half-built
index (or a missing `bm25s` dependency) never sinks a live turn — it just falls back to live.

## Compliance

The local Wikipedia dump is **RAW, non-generated content** (the course explicitly suggests a Wikipedia
dump for RAG). Name it in the video: **"local full-enwiki dump + BM25 (bm25s)"**, with live Wikipedia as
the on-miss fallback. The free Guardian/News sources for the News competition are named separately.
