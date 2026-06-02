#!/usr/bin/env bash
# Build the local full-enwiki BM25 index, end to end. Run from the repo root:
#     bash scripts/build_enwiki_bm25.sh
# In a Colab cell:   !bash scripts/build_enwiki_bm25.sh
#
# Steps: download dump -> wikiextractor (clean) -> prepare_enwiki (flatten) -> bm25_index (build) -> smoke query.
# Each step is SKIPPED if its output already exists, so a re-run resumes where it stopped.
#
# RESOURCES (full enwiki): ~80 GB free disk, and the bm25s BUILD wants ~32-64 GB RAM. Free Colab (~12-25 GB)
# will likely OOM on the full build -> either build on a high-RAM machine and HOST the index dir (Drive/HF),
# or set MAX_DOCS below to cap it (a smaller, partial index that still loads/queries the same way).
#
# Knobs (env vars): WORK (scratch dir), OUT (index dir), MAX_DOCS (cap for a smoke/partial build), PROCS.
set -euo pipefail

WORK="${WORK:-enwiki_build}"
OUT="${OUT:-data/corpus/enwiki/bm25_index}"
PROCS="${PROCS:-4}"
MAX_DOCS="${MAX_DOCS:-}"                       # e.g. MAX_DOCS=20000 for a quick smoke build; empty = full.
DUMP_URL="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"
DUMP="${WORK}/enwiki-latest-pages-articles-multistream.xml.bz2"
EXTRACT="${WORK}/extracted"
ARTICLES="${WORK}/articles.jsonl"

mkdir -p "${WORK}"
export PYTHONPATH="src:${PYTHONPATH:-}"        # so `python -m retrieval.*` resolves.

echo "==> [0/4] deps (bm25s + wikiextractor)"
python -c "import bm25s" 2>/dev/null        || pip -q install bm25s
python -c "import wikiextractor" 2>/dev/null || pip -q install wikiextractor

echo "==> [1/4] download dump (~22 GB)  -> ${DUMP}"
if [ ! -f "${DUMP}" ]; then wget -c -O "${DUMP}" "${DUMP_URL}"; else echo "    (exists, skip)"; fi

echo "==> [2/4] wikiextractor clean -> ${EXTRACT}  (this takes a while)"
if [ ! -d "${EXTRACT}" ]; then
    python -m wikiextractor.WikiExtractor "${DUMP}" --json --no-templates --processes "${PROCS}" -o "${EXTRACT}"
else echo "    (exists, skip)"; fi

echo "==> [3/4] flatten -> ${ARTICLES}"
if [ ! -f "${ARTICLES}" ]; then
    python -m retrieval.prepare_enwiki --input "${EXTRACT}" --out "${ARTICLES}" \
        ${MAX_DOCS:+--max-docs "${MAX_DOCS}"}
else echo "    (exists, skip)"; fi

echo "==> [4/4] build BM25 index -> ${OUT}  (RAM-heavy)"
python -m retrieval.bm25_index --corpus "${ARTICLES}" --out "${OUT}" ${MAX_DOCS:+--max-docs "${MAX_DOCS}"}

echo "==> smoke query"
python - <<'PY'
import os, sys; sys.path.insert(0, "src")
from retrieval.bm25_retriever import BM25Retriever
from schemas import Question
r = BM25Retriever(index_dir=os.environ.get("OUT", "data/corpus/enwiki/bm25_index"), top_k=2)
q = Question(qid="t", text="In which U.S. state was 'The Shawshank Redemption' filmed?",
             options={"A": "California", "B": "Ohio", "C": "Maine", "D": "New York"})
docs = r.retrieve(q)
print("  top:", docs[0].doc_id if docs else "(none)")
print("  has 'Ohio':", any("Ohio" in d.text for d in docs))
PY

echo ""
echo "DONE. Index at: ${OUT}"
echo "Now set in configs/live.yaml ->  retrieval.bm25_index_path: ${OUT}"
echo "(Portable: host this dir and point bm25_index_path at the copy on Colab -- no rebuild.)"
