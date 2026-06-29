"""model.py — RAGCopilot dual-source retrieval + evolving memory.

Implements a MEMCoder-style architecture (arXiv:2604.24222, April 2026) for
private-corpus RAG:

  1. **Static source** — a TF-IDF retriever over the bundled enterprise
     knowledge base (corpus of *ML / RAG / finance-risk / medical / MLOps*
     notes shipped in ``data/raw/``).
  2. **Evolving memory source** — a per-session memory buffer that captures
     past Q&A, retrieved passages, and feedback; queries against the memory
     are answered with the same TF-IDF index but with *recency-decayed* term
     frequencies.
  3. **Reciprocal Rank Fusion** combines the two ranked lists into a final
     retrieval ordering, mirroring the dual-source retrieval pattern from
     MEMCoder's *static documentation + historical guidelines* pipeline.

Public API
----------
``fit(docs, sources)`` builds the static index; ``add_memory(question, answer,
passages)`` records an entry to the evolving memory; ``retrieve(query, k=3)``
returns fused (doc_index, score, passage) tuples.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Memory record utilities
# ─────────────────────────────────────────────────────────────────────────────
class MemoryEntry:
    __slots__ = ("question", "answer", "passages", "ts", "use_count")

    def __init__(self, question: str, answer: str, passages: list[str]) -> None:
        self.question = question
        self.answer = answer
        self.passages = passages
        self.ts = time.time()
        self.use_count = 0

    def to_text(self) -> str:
        """Memory entry serialised for retrieval — question + answer."""
        return f"Q: {self.question}\nA: {self.answer}"


# ─────────────────────────────────────────────────────────────────────────────
# Sparse retriever — TF-IDF over the union of static and memory texts
# ─────────────────────────────────────────────────────────────────────────────
class DualSourceRetriever:
    """TF-IDF retriever fused between a static corpus and evolving memory.

    Parameters
    ----------
    static_docs : list[str]
        Base corpus (chunked + uploaded).
    static_sources : list[str]
        Display source for each static doc.
    decay_tau_sec : float
        Time constant for memory recency weighting; older entries are
        down-weighted exponentially.
    """

    def __init__(self, static_docs: list[str], static_sources: list[str],
                 decay_tau_sec: float = 86_400.0 * 7) -> None:
        self.static_docs = list(static_docs)
        self.static_sources = list(static_sources)
        self.memory: list[MemoryEntry] = []
        self.decay_tau_sec = decay_tau_sec
        self._build_static_index()
        self._rf_k = 60.0  # RRF constant

    # ── index management ─────────────────────────────────────────────────────
    def _build_static_index(self) -> None:
        self._static_index = _build_tfidf(self.static_docs)

    def _memory_recency_weights(self, now: float) -> np.ndarray:
        if not self.memory:
            return np.zeros(0)
        ts = np.asarray([m.ts for m in self.memory])
        delta = np.maximum(0.0, now - ts)
        return np.exp(-delta / self.decay_tau_sec)

    def add_memory(self, question: str, answer: str, passages: list[str]) -> None:
        if not (question and answer):
            return
        self.memory.append(MemoryEntry(question, answer, passages))

    def add_memory_doc(self, question: str, answer: str, passages: list[str]) -> int:
        """Append a memory entry; returns the new memory-doc count."""
        self.add_memory(question, answer, passages)
        return len(self.memory)

    # ── retrieval (dual-source + RRF fusion) ─────────────────────────────────
    def retrieve(self, query: str, k: int = 3, now: float | None = None) -> list[dict[str, Any]]:
        if now is None:
            now = time.time()
        static_results = _retrieve(self._static_index, query, k=k * 3)

        if self.memory:
            memory_docs = [m.to_text() for m in self.memory]
            memory_index = _build_tfidf(memory_docs)
            memory_scores = _score(memory_index, query)
            recency = self._memory_recency_weights(now)
            memory_scores *= 1.0 + recency  # boost recent entries
            top = np.argsort(-memory_scores)[:k * 3]
            memory_results = [
                {
                    "memory_idx": int(j),
                    "memory_score": float(memory_scores[j]),
                    "passages": self.memory[j].passages,
                    "memory_question": self.memory[j].question,
                    "memory_answer": self.memory[j].answer,
                }
                for j in top
                if memory_scores[j] > 0
            ]
        else:
            memory_results = []

        # Reciprocal Rank Fusion over static passages and memory passages
        # (each memory entry's passages are surfaced as candidates).
        fused: dict[int, dict[str, Any]] = {}
        for rank, (idx, sc, passage) in enumerate(static_results, start=1):
            fused.setdefault(idx, {"doc_idx": idx, "score": 0.0, "passage": passage,
                                    "static_rank": rank, "memory_rank": None})
            fused[idx]["score"] += 1.0 / (self._rf_k + rank)
        for rank, m in enumerate(memory_results, start=1):
            for passage in m["passages"]:
                # Match by passage text (statics were deduped during index build)
                idx_in_static = self._passage_index(passage)
                if idx_in_static is None:
                    continue
                fused.setdefault(idx_in_static, {
                    "doc_idx": idx_in_static, "score": 0.0,
                    "passage": passage, "static_rank": None, "memory_rank": rank,
                })
                fused[idx_in_static]["score"] += 1.0 / (self._rf_k + rank)
                if fused[idx_in_static]["memory_rank"] is None:
                    fused[idx_in_static]["memory_rank"] = rank

        ranked = sorted(fused.values(), key=lambda x: -x["score"])[:k]
        return ranked

    def _passage_index(self, passage: str) -> int | None:
        for i, doc in enumerate(self.static_docs):
            if doc == passage:
                return i
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sparse TF-IDF primitives
# ─────────────────────────────────────────────────────────────────────────────
def _idf_matrix(token_lists: list[list[str]]) -> tuple[dict[str, float], list[str]]:
    df: Counter = Counter()
    for tokens in token_lists:
        for w in set(tokens):
            df[w] += 1
    n = max(1, len(token_lists))
    idf = {w: math.log((1 + n) / (1 + d)) + 1.0 for w, d in df.items()}
    keys = list(idf.keys())
    ix = {w: i for i, w in enumerate(keys)}
    return {"idf": idf, "ix": ix, "keys": keys}, ix


def _build_tfidf(docs: list[str]) -> dict[str, Any]:
    token_lists = [_tokenize(d) for d in docs]
    inv, ix = _idf_matrix(token_lists)
    n = len(docs)
    V = len(ix)
    M = np.zeros((n, V), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        if not tokens:
            continue
        c: Counter = Counter(tokens)
        for w, cnt in c.items():
            j = ix.get(w)
            if j is not None:
                M[i, j] = cnt * inv["idf"][w]
        norm = np.linalg.norm(M[i])
        if norm > 0:
            M[i] /= norm
    return {"M": M, "idf": inv["idf"], "ix": ix, "docs": docs, "n_docs": n}


def _score(model: dict[str, Any], query: str) -> np.ndarray:
    V = len(model["ix"])
    vec = np.zeros(V, dtype=np.float32)
    for w in _tokenize(query):
        j = model["ix"].get(w)
        if j is not None:
            vec[j] += model["idf"][w]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return model["M"] @ vec


def _retrieve(model: dict[str, Any], query: str, k: int = 3) -> list[tuple[int, float, str]]:
    scores = _score(model, query)
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i]), model["docs"][i]) for i in order]


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers used by app.py — kept 100 % backwards-compatible
# ─────────────────────────────────────────────────────────────────────────────
def build_tfidf(docs: list[str]) -> dict[str, Any]:
    return _build_tfidf(docs)


def score(model: dict[str, Any], query: str) -> np.ndarray:
    return _score(model, query)


def retrieve(model: dict[str, Any], query: str, k: int = 3) -> list[tuple[int, float, str]]:
    return _retrieve(model, query, k=k)


def fit(docs: list[str], sources: list[str], decay_tau_sec: float | None = None) -> DualSourceRetriever:
    """Convenience wrapper mirroring the ``fit_and_evaluate`` convention."""
    rs = DualSourceRetriever.__new__(DualSourceRetriever)
    rs.static_docs = list(docs)
    rs.static_sources = list(sources)
    rs.memory = []
    rs.decay_tau_sec = decay_tau_sec if decay_tau_sec is not None else 86_400.0 * 7
    rs._rf_k = 60.0
    rs._build_static_index()
    return rs


def fit_and_evaluate(docs: list[str], sources: list[str],
                     queries: list[tuple[str, int]],
                     ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    """Run dual-source retrieval on the static corpus and compute Recall@k + MRR."""
    model = fit(docs, sources)
    return evaluate_retrieval(model, queries, ks=ks)


def evaluate_retrieval(model: DualSourceRetriever,
                       queries: list[tuple[str, int]],
                       ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    metrics: dict[str, Any] = {"n_queries": len(queries), "method": "dual_source_rrf"}
    if not queries:
        for k_ in ks:
            metrics[f"recall_at_{k_}"] = 0.0
        metrics["mrr"] = 0.0
        return metrics

    max_k = max(ks)
    hits = {k_: 0 for k_ in ks}
    rr = 0.0
    latencies: list[float] = []
    for q, gold in queries:
        t0 = time.perf_counter()
        results = model.retrieve(q, k=max_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        for k_ in ks:
            if any(r["doc_idx"] == gold for r in results[:k_]):
                hits[k_] += 1
        for rank, r in enumerate(results, start=1):
            if r["doc_idx"] == gold:
                rr += 1.0 / rank
                break

    for k_ in ks:
        metrics[f"recall_at_{k_}"] = hits[k_] / len(queries)
    metrics["mrr"] = rr / len(queries)
    metrics["avg_latency_ms"] = float(np.mean(latencies))
    metrics["memory_size"] = len(model.memory)
    return metrics
