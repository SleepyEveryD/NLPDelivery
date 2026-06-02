# Who Wants to Be a PoliMillionaire? — Local-LLM Quiz Agent

NLP 2025/26 Group Assignment. A chatbot that plays the online quiz **"Who Wants to Be a
PoliMillionaire?"** using **open-weight LLMs run locally** (on Google Colab), answering each
question within the **30-second** limit.

> Code comments throughout are written in **Yoda speaking style** (an assignment requirement).

## Constraints (from the assignment)
- Models run **locally** — no OpenAI / external LLM APIs.
- Open-weight models only.
- ~30 s per question.
- RAG allowed using **raw retrieved content only** (HTML/PDF/text), never generated answers.
- Deliverable: a self-explanatory Colab **notebook** (+ `.html`) and a 5-min video.
- Due **2026-06-02 23:00** (WeBeep).

## Target stack
Google Colab (T4/L4) · `Qwen/Qwen2.5-7B-Instruct` 4-bit (bitsandbytes) · `transformers` + `accelerate`
· optional FAISS + `sentence-transformers` RAG · optional calculator tool · optional multi-model voting.

## Layout
```
src/         modular library (imported into notebooks)
  schemas.py     shared data contract (Question, Prediction, EvalRecord, ...)
  config.py      YAML -> typed RunConfig
  game/          adapter to the quiz text API (Tutorial 7)  [contract TBD]
  inference/     LLMEngine (transformers backend)
  prompting/     PromptBuilder (zero/few-shot, CoT)
  classify/      QuestionClassifier (routing)
  retrieval/     Retriever (FAISS / raw-content search)
  tools/         calculator (safe AST eval)
  agent/         QAPipeline (orchestrator) + voting
  evaluation/    logger (JSONL) + metrics + benchmark runner
  utils/         timing / latency guard
configs/     YAML run configs
notebooks/   narrative experiments (the graded deliverable)
experiments/ per-run logs (JSONL + meta.json)  [gitignored]
data/        corpus + dev question sets          [gitignored]
memory/      project memory (architecture, decisions, experiments, tasks, ...)
```

## Run on Colab (intended flow)
```python
!git clone <repo-url> /content/NLP
%cd /content/NLP
!pip install -r requirements.txt
import sys; sys.path.insert(0, "src")          # the library importable, this makes
from config import RunConfig
cfg = RunConfig.from_yaml("configs/base.yaml")
# ... build engine + pipeline, then benchmark or play the game ...
```

## Project memory
Read `memory/` before making changes — it preserves the architecture, decisions, the roadmap,
experiment results, and known errors across sessions. Start with `memory/architecture.md` and
`memory/tasks.md`.

## Status
Phase 0 (scaffolding) done. Phase 1 (baseline single-model QA) is next. The game API client is a
documented stub pending the WeBeep Tutorial 7 notebook — see `memory/tasks.md` item **[B1]**.
