"""LLM inference engine abstraction. Pluggable models, this enables.

Swap Qwen for Mistral or Gemma without touching the pipeline, we can -- minimal coupling, the goal is.
The Colab default: Qwen2.5-7B-Instruct in 4-bit (bitsandbytes), via transformers + accelerate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

# Heavy GPU libraries, import here we attempt -- on non-GPU machines, fail gracefully they may.
# Only when TransformersEngine is instantiated, the real loading occurs.
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # On CPU-only dev boxes, missing these is fine -- not instantiated, the class is.
    _TRANSFORMERS_AVAILABLE = False


class LLMEngine(ABC):
    """The contract every model backend must honor, this is."""

    name: str = "unknown"

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        """Text in, text out. Load the model once and reuse it many times, we do."""
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Pay the first-call cost before the timer starts, this lets us."""
        ...


class TransformersEngine(LLMEngine):
    """Qwen2.5-7B 4-bit on a Colab T4/L4, this runs. (Implementation: Phase 1.)

    Once loaded, the tokenizer and model live in memory -- swapped between calls, they are not.
    Apply the chat template here we do; the PromptBuilder only supplies the user-turn content.
    """

    def __init__(self, model_name: str, quantization: str = "4bit", dtype: str = "bfloat16"):
        """Once, the model we load -- reuse it many times, we do.

        Args:
            model_name:    The HuggingFace model ID, this is (e.g. "Qwen/Qwen2.5-7B-Instruct").
            quantization:  "4bit" for nf4 bitsandbytes quant; None/"none"/"fp16"/"bf16" for full dtype.
            dtype:         "bfloat16" or "float16" -- the compute dtype, this is.
        """
        # Available, the transformers stack must be -- only on Colab, instantiated this class is.
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "torch and transformers, installed they are not. "
                "On Colab, run this you must -- not on a local CPU-only box."
            )

        # The model name, stored it is -- used for logging and EvalRecord.model, it will be.
        self.name: str = model_name

        # Token counts from the last generate() call, tracked here they are -- consumers read them.
        self.last_tokens_in: int = 0
        self.last_tokens_out: int = 0

        # Map dtype string to torch dtype -- "bfloat16" is the default, matches D-002 and the course recipe.
        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16":     torch.bfloat16,
            "float16":  torch.float16,
            "fp16":     torch.float16,
        }
        torch_dtype = _dtype_map.get(dtype, torch.bfloat16)

        # The quantization config, built only for 4-bit -- matches exactly the course recipe (D-012).
        # nf4 + double quant + bf16 compute → ~5–6 GB VRAM, fits T4 it does.
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # Always bf16 compute for nf4, the course mandates.
            )
        else:
            # Full-precision or fp16 path -- for machines with more VRAM, useful this is.
            quant_config = None

        # The tokenizer, loaded first we do -- lightweight, it is.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # The model, loaded with device_map="auto" -- distributes across available GPUs, accelerate does.
        # quantization_config=None passes cleanly when full precision chosen is.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
        )

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        """From a user-turn string, a response string we produce.

        The chat template, applied here it is -- the PromptBuilder provides raw user content only.
        Greedy decoding by default, temperature=0 means -- deterministic and fast, this keeps things.

        Args:
            prompt:         The user-turn content (raw text, no template applied yet), this is.
            max_new_tokens: How many new tokens to generate at most, this caps.
            temperature:    0.0 for greedy; >0 for sampling -- D-004 mandates 0 as default.

        Returns:
            The decoded model response, stripped of special tokens and whitespace, this is.
        """
        # Wrap the raw prompt in a chat message -- the model expects a structured conversation, it does.
        messages = [{"role": "user", "content": prompt}]

        # Apply the chat template -- a formatted string with special tokens, this produces.
        # tokenize=False: a string we want, not token IDs -- tokenized separately, it is.
        text: str = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize and move to model device -- the model's GPU, the tensors must live on.
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # The number of input tokens, recorded before generation -- needed for EvalRecord.tokens_in.
        self.last_tokens_in = inputs["input_ids"].shape[-1]

        # Generation kwargs, set by temperature -- greedy (do_sample=False) is D-004's default.
        if temperature == 0.0:
            gen_kwargs = {"do_sample": False}
        else:
            gen_kwargs = {"do_sample": True, "temperature": temperature}

        # Inference mode, used here we do -- no gradient graph built, memory and speed saved are.
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )

        # Only the newly generated tokens we decode -- the prompt echo, skip it we must.
        # output_ids shape: (batch=1, total_seq_len); slice from input length to end, we do.
        new_token_ids = output_ids[0, self.last_tokens_in:]

        # The count of new tokens, store we do -- exposed as last_tokens_out for callers.
        self.last_tokens_out = new_token_ids.shape[-1]

        # Decode to string, skip special tokens -- clean text the caller receives.
        decoded: str = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        # Whitespace at both ends, removed it is -- model sometimes adds trailing newlines.
        return decoded.strip()

    def warmup(self) -> None:
        """The first-call JIT / CUDA compilation cost, pay it here we do.

        Before the 30-second game timer starts, call this -- cold-start latency, absorbed upfront it is.
        Errors, swallowed silently they are -- warmup failure must not crash the game.
        """
        # One tiny generation, run we do -- kernels compiled and caches warmed, after this they are.
        try:
            self.generate("Hello", max_new_tokens=1)
        except Exception:
            # Warmup failure, ignore we do -- log nothing, so the caller's flow uninterrupted stays.
            pass


# ===========================================================================
# A deterministic simulated engine -- the adaptive-routing harness, WITHOUT a GPU it lets us run.
#
# !!! NOT a language model. A FIXTURE, this is. !!!
# Its correctness comes from a hand-set skill table that ENCODES the experiment's hypothesis (structured
# enumeration wins on counting; direct answering wins on recall; checklists win on logic). So any plot it
# produces validates the PLUMBING of the experiment -- the routing, logging, metrics and figures -- and
# NOT the real model. The real numbers, only the Colab Qwen run (TransformersEngine) yields them.
# Use it for: local smoke tests, CI, and demonstrating the analysis end-to-end before the GPU run.
# ===========================================================================

import hashlib
import re as _re

from schemas import Question  # local import -- the heavy engine path above does not need it.


# The hypothesis, as P(correct | true_category, strategy) in [0,1].
# Read each ROW as "for this kind of reasoning, which prompt helps?" -- the diagonal of specialised
# strengths (interval->structured, factual->direct, logic->checklist) is the whole bet, made explicit.
_SIM_SKILL: dict[str, dict[str, float]] = {
    # category                direct generic structured checklist few_shot(universal)
    "arithmetic":            {"direct_answer": 0.45, "generic_cot": 0.80, "structured_enumeration_cot": 0.70, "checklist_cot": 0.72, "few_shot_v1": 0.65},
    "temporal_reasoning":    {"direct_answer": 0.40, "generic_cot": 0.58, "structured_enumeration_cot": 0.80, "checklist_cot": 0.60, "few_shot_v1": 0.50},
    "interval_counting":     {"direct_answer": 0.25, "generic_cot": 0.45, "structured_enumeration_cot": 0.85, "checklist_cot": 0.55, "few_shot_v1": 0.40},
    "discrete_enumeration":  {"direct_answer": 0.35, "generic_cot": 0.55, "structured_enumeration_cot": 0.82, "checklist_cot": 0.60, "few_shot_v1": 0.50},
    "factual_qa":            {"direct_answer": 0.82, "generic_cot": 0.70, "structured_enumeration_cot": 0.55, "checklist_cot": 0.62, "few_shot_v1": 0.80},
    "commonsense":           {"direct_answer": 0.78, "generic_cot": 0.68, "structured_enumeration_cot": 0.55, "checklist_cot": 0.65, "few_shot_v1": 0.74},
    "logical_reasoning":     {"direct_answer": 0.45, "generic_cot": 0.62, "structured_enumeration_cot": 0.58, "checklist_cot": 0.80, "few_shot_v1": 0.58},
    "multi_hop":             {"direct_answer": 0.40, "generic_cot": 0.60, "structured_enumeration_cot": 0.55, "checklist_cot": 0.78, "few_shot_v1": 0.55},
}

# Typical generated length per strategy (in "tokens"), the simulator uses -- so the reasoning-length and
# latency figures show the real trade-off: verbose chains cost time. (~11 tok/s, the Qwen-7B-4bit rate.)
_SIM_TOKENS: dict[str, int] = {
    "direct_answer": 4,
    "generic_cot": 55,
    "structured_enumeration_cot": 140,
    "checklist_cot": 115,
    "few_shot_v1": 8,
}
_SIM_TOK_PER_S: float = 11.0   # Decode speed the latency simulation assumes (the documented Qwen-4bit rate).

# Prompt-directive signatures -> the canonical strategy name (for the skill-table lookup).
_SIM_STRATEGY_SIGNATURES: list[tuple[str, str]] = [
    ("EXPLICIT ENUMERATION", "structured_enumeration_cot"),
    ("Work through this checklist", "checklist_cot"),
    ("Answer immediately with ONLY the letter", "direct_answer"),
    ("Reply with ONLY the letter", "direct_answer"),
    ("Let's think step by step", "generic_cot"),
    ("Think step by step briefly", "generic_cot"),
    ("Solve in AT MOST 3", "generic_cot"),       # cot_v2 family -> treat as a generic chain.
    ("Answer with ONLY the letter", "few_shot_v1"),
]


class SimulatedReasoningEngine(LLMEngine):
    """A deterministic stand-in for the LLM -- the routing harness end-to-end, locally it exercises.

    Given the dataset up front (so it knows each question's gold letter and TRUE reasoning category), it:
      * reads the strategy from the prompt's directives,
      * looks up P(correct | true_category, strategy) in `_SIM_SKILL`,
      * decides correctness DETERMINISTICALLY (a stable hash of the qid, not RNG -- reproducible runs),
      * emits strategy-flavoured text of a realistic length, and a simulated decode latency.

    A fixture it is, not a model -- see the module banner above. `name` says so, loudly.
    """

    name = "SimulatedReasoningEngine(FIXTURE-not-a-real-model)"

    def __init__(self, questions: list[Question], categories: dict[str, str]):
        """
        Args:
            questions:  the eval set -- gold letters and a text->qid index, from these we build.
            categories: qid -> TRUE reasoning-category label (drives the skill table).
        """
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        self.last_latency_s = 0.0   # The runner prefers this over wall-clock when present (a fixture, we are).
        self._gold: dict[str, str] = {q.qid: (q.gold or "A") for q in questions}
        self._opts: dict[str, list[str]] = {q.qid: sorted(q.options.keys()) for q in questions}
        self._cat: dict[str, str] = dict(categories)
        # A normalised question-text -> qid index, so from the prompt the question we can recover.
        self._text_to_qid: dict[str, str] = {self._norm(q.text): q.qid for q in questions}

    @staticmethod
    def _norm(text: str) -> str:
        # Lowercase, collapse whitespace -- a stable lookup key from a question's text, this makes.
        return _re.sub(r"\s+", " ", (text or "").strip().lower())

    def _strategy_of(self, prompt: str) -> str:
        for needle, strat in _SIM_STRATEGY_SIGNATURES:
            if needle in prompt:
                return strat
        return "generic_cot"   # Unknown directives -> a plain chain, assume we do.

    def _qid_of(self, prompt: str) -> str | None:
        # The LAST "Question:" line is the real one (few-shot prepends exemplars before it).
        matches = _re.findall(r"Question:\s*(.+)", prompt)
        if not matches:
            return None
        return self._text_to_qid.get(self._norm(matches[-1]))

    @staticmethod
    def _score(qid: str, strategy: str) -> float:
        # A stable pseudo-score in [0,1) -- md5 over (qid, strategy), so process-independent it is.
        h = hashlib.md5(f"{qid}|{strategy}".encode()).hexdigest()
        return (int(h[:8], 16) % 10_000) / 10_000.0

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        strategy = self._strategy_of(prompt)
        qid = self._qid_of(prompt)

        self.last_tokens_in = len(prompt.split())
        n_out = _SIM_TOKENS.get(strategy, 40)
        self.last_tokens_out = n_out
        self.last_latency_s = n_out / _SIM_TOK_PER_S

        # Unknown question (text not in the index) -> a harmless default, return we do.
        if qid is None:
            return "Answer: A"

        gold = self._gold.get(qid, "A")
        cat = self._cat.get(qid, "commonsense")
        skill = _SIM_SKILL.get(cat, {}).get(strategy, 0.5)
        correct = self._score(qid, strategy) < skill

        if correct:
            letter = gold
        else:
            # A deterministic WRONG letter -- the next available option after gold, wrapping around.
            opts = self._opts.get(qid) or ["A", "B", "C", "D"]
            others = [o for o in opts if o != gold] or ["A"]
            letter = others[int(self._score(qid, strategy) * len(others)) % len(others)]

        # Strategy-flavoured filler, so the reasoning-LENGTH metric reflects the real verbosity gap.
        body = {
            "direct_answer": "",
            "few_shot_v1": "",
            "generic_cot": "Step 1: consider the question.\nStep 2: reason it through.\n",
            "structured_enumeration_cot": (
                "Enumerate:\n- item 1\n- item 2\n- item 3\nBoundary check: endpoints inside range.\n"
                "Running total computed.\n"
            ),
            "checklist_cot": (
                "1. Restated.\n2. Assumptions listed.\n3. Each option evaluated.\n4. Validated against details.\n"
            ),
        }.get(strategy, "")
        return f"{body}Answer: {letter}"

    def warmup(self) -> None:
        # A fixture -- nothing to warm. A no-op, this is.
        return None
