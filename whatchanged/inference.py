from __future__ import annotations
import json
import re
from typing import Protocol
from whatchanged.models import DOMAINS, EVENTS, Finding


class LLMBackend(Protocol):
    def complete(self, prompt: str, max_tokens: int = 256,
                 grammar: str | None = None, temperature: float = 0.2) -> str: ...


class FakeBackend:
    """Test/dev backend that returns a fixed string and records the last prompt."""
    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    def complete(self, prompt: str, max_tokens: int = 256,
                 grammar: str | None = None, temperature: float = 0.2) -> str:
        self.last_prompt = prompt
        return self.response


def parse_extraction(raw: str) -> dict:
    """Robustly pull the first JSON object out of model output and coerce it to the
    schema: domains clamped 1..5, events clamped >=0 ints, unknown keys dropped."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}
    out: dict = {}
    for k, v in data.items():
        if k in DOMAINS:
            try:
                out[k] = max(1, min(5, int(v)))
            except (TypeError, ValueError):
                continue
        elif k in EVENTS:
            try:
                out[k] = max(0, int(v))
            except (TypeError, ValueError):
                continue
    return out


def build_extract_prompt(note: str) -> str:
    return (
        "You convert a caregiver's free-text note about an elderly person's day into "
        "JSON. Only output a JSON object. Rate each domain 1-5 (5 = best) ONLY if the "
        "note mentions it; omit domains not mentioned. Count events as non-negative "
        "integers.\n"
        f"Domains: {', '.join(DOMAINS)}\n"
        f"Events: {', '.join(EVENTS)}\n"
        "Do not diagnose. Output JSON only.\n\n"
        f'Note: "{note}"\n'
        "JSON:"
    )


def build_extract_grammar() -> str:
    """GBNF that constrains decoding to a JSON object whose keys are EXACTLY the schema
    fields, each mapped to an integer. This guarantees valid, parseable output and makes
    out-of-schema keys (e.g. a slangy 'tumble' or 'freezes') impossible at decode time."""
    keys = " | ".join(r'"\"' + k + r'\""' for k in (*DOMAINS, *EVENTS))
    return (
        'root ::= "{" ws ( pair ( ws "," ws pair )* )? ws "}"\n'
        'pair ::= key ws ":" ws [0-9]+\n'
        "key ::= " + keys + "\n"
        "ws ::= [ \\t\\n]*\n"
    )


EXTRACT_GRAMMAR = build_extract_grammar()


def extract_note(note: str, backend: LLMBackend) -> dict:
    if not note.strip() or backend is None:
        return {}
    # Grammar-constrained, greedy: always valid schema JSON, deterministic (matches eval).
    raw = backend.complete(build_extract_prompt(note), max_tokens=128,
                           grammar=EXTRACT_GRAMMAR, temperature=0.0)
    return parse_extraction(raw)


def build_narrate_prompt(findings: list[Finding]) -> str:
    bullet = "\n".join(f"- {f.summary}" for f in findings)
    return (
        "You are writing a short, plain-language paragraph for a caregiver to bring to "
        "a doctor. Summarize ONLY the findings below. Be factual and descriptive. Do "
        "NOT diagnose, predict, or recommend treatment.\n\n"
        f"Findings:\n{bullet}\n\nParagraph:"
    )


def narrate_findings(findings: list[Finding], backend: LLMBackend) -> str:
    if not findings:
        return "No notable changes were detected in the logged period."
    fallback = "Over the logged period: " + "; ".join(f.summary for f in findings) + "."
    if backend is None:
        return fallback
    try:  # never let a slow/failed LLM call break the report
        text = backend.complete(build_narrate_prompt(findings), max_tokens=140).strip()
        return text or fallback
    except Exception:  # noqa: BLE001
        return fallback


class LlamaCppBackend:
    """Wraps a local GGUF model via llama-cpp-python. Model loads lazily on first
    use so importing this module never requires the model or the native lib."""
    def __init__(self, model_path: str, n_ctx: int = 1024, n_threads: int | None = None):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None

    def _ensure(self):
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self.model_path, n_ctx=self.n_ctx,
                n_threads=self.n_threads, verbose=False,
            )
        return self._llm

    def complete(self, prompt: str, max_tokens: int = 256,
                 grammar: str | None = None, temperature: float = 0.2) -> str:
        llm = self._ensure()
        g = None
        if grammar is not None:
            from llama_cpp import LlamaGrammar
            g = LlamaGrammar.from_string(grammar, verbose=False)
        out = llm.create_completion(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            stop=["\n\n\n"], grammar=g,
        )
        return out["choices"][0]["text"]
