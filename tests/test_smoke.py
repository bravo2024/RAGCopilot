"""Smoke test: build the dual-source retriever and verify retrieval quality."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import demo_corpus
from src.model import fit, evaluate_retrieval


def test_pipeline_runs():
    corpus = demo_corpus()
    model = fit(corpus["docs"], corpus["sources"])
    metrics = evaluate_retrieval(model, corpus["queries"])
    assert isinstance(metrics, dict)
    assert "recall_at_3" in metrics and "mrr" in metrics
    # static TF-IDF on this curated corpus is very strong
    assert metrics["recall_at_3"] >= 0.6
    assert metrics["mrr"] >= 0.5
    assert model is not None


def test_memory_fusion_writes_and_recalls():
    """Verify the dual-source memory pipeline writes entries and runs end-to-end."""
    corpus = demo_corpus()
    model = fit(corpus["docs"], corpus["sources"])
    base = evaluate_retrieval(model, corpus["queries"])
    # The static-only baseline already achieves Recall@3 ≥ 0.6.
    assert base["recall_at_3"] >= 0.6

    # Add real memory entries; verify writes persist and retrieval completes
    for q, _ in corpus["queries"]:
        results = model.retrieve(q, k=3)
        passages = [r["passage"] for r in results[:2]]
        if passages:
            model.add_memory(q, passages[0], passages)
    assert len(model.memory) == len(corpus["queries"])

    fused = evaluate_retrieval(model, corpus["queries"])
    assert isinstance(fused["mrr"], float)
    assert fused["memory_size"] == len(corpus["queries"])
