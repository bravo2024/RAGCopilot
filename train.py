from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag import demo_corpus, save_metrics, print_report
from src.model import fit, evaluate_retrieval


def main() -> None:
    corpus = demo_corpus()
    docs = corpus["docs"]
    sources = corpus["sources"]
    queries = corpus["queries"]

    print(f"Loaded {len(docs)} docs across {len(set(sources))} sources")
    print(f"Eval set: {len(queries)} queries")

    # Dual-source retrieval (static only; memory starts empty)
    model = fit(docs, sources)
    metrics = evaluate_retrieval(model, queries)
    print_report({"Retrieval (static)": metrics})

    # Simulate a memory feedback pass — store Q/A pairs and re-evaluate
    # the same queries to demonstrate dual-source fusion.
    for q, gold in queries:
        results = model.retrieve(q, k=3)
        passages = [r["passage"] for r in results[:2]]
        if passages:
            answer = passages[0][:160]
            model.add_memory(q, answer, passages)
    metrics2 = evaluate_retrieval(model, queries)
    print_report({"Retrieval (static + memory)": metrics2})

    final = {
        "method": "dual_source_rrf",
        "arxiv": "2604.24222 (MEMCoder)",
        "n_docs": len(docs),
        "n_queries": len(queries),
        "n_sources": len(set(sources)),
        "static_only": {
            "recall_at_1": metrics["recall_at_1"],
            "recall_at_3": metrics["recall_at_3"],
            "recall_at_5": metrics["recall_at_5"],
            "mrr": metrics["mrr"],
            "avg_latency_ms": metrics["avg_latency_ms"],
        },
        "with_memory": {
            "recall_at_1": metrics2["recall_at_1"],
            "recall_at_3": metrics2["recall_at_3"],
            "recall_at_5": metrics2["recall_at_5"],
            "mrr": metrics2["mrr"],
            "memory_size": metrics2["memory_size"],
        },
    }

    out_dir = Path(__file__).parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(final, str(out_dir / "metrics.json"))
    print(f"\nMetrics saved to {out_dir / 'metrics.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
