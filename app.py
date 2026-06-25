# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag import (
    ExtractiveGenerator,
    NVIDIA_NIM_MODELS,
    NvidiaNIMGenerator,
    build_retriever,
    demo_corpus,
    evaluate_retrieval,
    faithfulness_score,
    read_uploads,
    retrieve,
    run_generator,
    save_retriever,
    token_overlap_f1,
)


def _fill_question(q):
    st.session_state.ask_question = q


st.set_page_config(
    page_title="RAGCopilot",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 55%, #0d9488 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { margin-bottom: 0.2rem; }
    .hero p  { margin-bottom: 0; opacity: 0.92; }
    .answer {
        background: #ecfeff;
        border-left: 4px solid #0891b2;
        padding: 0.9rem 1rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .step {
        background: #f0fdfa;
        border-left: 4px solid #0d9488;
        padding: 0.75rem 1rem;
        border-radius: 0.25rem;
        margin: 0.4rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🔎 RAGCopilot</h1>
      <p>Streamlit RAG demo &middot; uploadable documents &middot; grounded answers &middot;
         Recall@k / MRR / faithfulness &middot; pluggable local LLM</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Corpus")
    include_demo = st.checkbox("Include built-in knowledge base", value=True)
    upload_col, clear_col = st.columns([3, 1])
    with upload_col:
        if "upload_key" not in st.session_state:
            st.session_state.upload_key = 0
        uploaded_files = st.file_uploader(
            "Upload .txt, .md, or .pdf",
            type=["txt", "md", "markdown", "pdf"],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.upload_key}",
        )
    with clear_col:
        st.write("")
        st.write("")
        if st.button("Clear", use_container_width=True, help="Remove uploaded files"):
            st.session_state.upload_key += 1
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) uploaded")
    st.divider()
    st.header("Retrieval")
    chunk_words = st.slider("Chunk size (words)", 60, 300, 120, step=20)
    overlap_words = st.slider("Chunk overlap (words)", 0, 80, 30, step=10)
    top_k = st.slider("Top-k passages", 1, 5, 3)
    st.markdown(f"**Backend:** TF-IDF (CPU)")
    st.divider()
    st.header("Generator")
    generator_choice = st.radio(
        "Answer generator",
        ["Extractive", "NVIDIA NIM"],
        index=1,
    )
    nim_model_label = None
    if generator_choice == "NVIDIA NIM":
        nim_model_label = st.selectbox(
            "NIM model",
            list(NVIDIA_NIM_MODELS.keys()),
            index=0,
        )
        st.caption(
            "Free tier at https://build.nvidia.com &middot; "
            "set `NVIDIA_API_KEY` in env or Streamlit secrets."
        )
    if st.button("Save retriever", use_container_width=True):
        save_retriever(st.session_state.retriever)
        st.success("Saved to models/model.pkl")


@st.cache_resource(show_spinner="Building retriever...")
def get_retriever(docs: tuple[str, ...]):
    return build_retriever(list(docs))


def build_corpus() -> tuple[list[str], list[str], list[tuple[str, int]]]:
    docs: list[str] = []
    sources: list[str] = []
    eval_queries: list[tuple[str, int]] = []

    if include_demo:
        demo = demo_corpus()
        docs.extend(demo["docs"])
        sources.extend(demo["sources"])
        eval_queries = list(demo["queries"])

    uploaded_docs, uploaded_sources = read_uploads(uploaded_files, chunk_words, overlap_words)
    docs.extend(uploaded_docs)
    sources.extend(uploaded_sources)
    return docs, sources, eval_queries


documents, source_labels, eval_queries = build_corpus()

if not documents:
    st.warning("Add the demo knowledge base or upload at least one text/PDF file.")
    st.stop()

retriever = get_retriever(tuple(documents))
st.session_state.retriever = retriever


def corpus_metrics() -> dict[str, Any]:
    """Return retrieval metrics for the labelled eval set, or zeros if unavailable."""
    if not eval_queries or not include_demo:
        return {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0}
    return evaluate_retrieval(retriever, eval_queries, ks=(1, 3, 5))


def avg_latency_ms() -> float:
    if not eval_queries or not include_demo:
        return 0.0
    return evaluate_retrieval(retriever, eval_queries, ks=(1, 3, 5)).get("avg_latency_ms", 0.0)


m = corpus_metrics()
cols = st.columns(5)
cols[0].metric("Chunks", f"{len(documents):,}")
cols[1].metric("Recall@3", f"{m['recall_at_3']:.1%}")
cols[2].metric("MRR", f"{m['mrr']:.2f}")
cols[3].metric("Retriever", "TF-IDF")
cols[4].metric("Latency", f"{avg_latency_ms():.1f} ms")


overview_tab, ask_tab, eval_tab, corpus_tab, faith_tab = st.tabs(
    ["Overview", "Ask", "Evaluate", "Corpus", "Faithfulness"]
)


SAMPLE_QUESTIONS = [
    "How does RAG reduce hallucination?",
    "What is U-Net used for in medical imaging?",
    "What is the difference between VaR and Expected Shortfall?",
    "How does cross-encoder reranking work?",
]


def _format_highlighted(text: str, query: str) -> str:
    import re
    q_tokens = set(re.findall(r"[a-z]+", query.lower()))
    out: list[str] = []
    for piece in re.split(r"(\s+)", text):
        if not piece:
            continue
        if piece.isspace():
            out.append(piece)
            continue
        words = re.findall(r"[a-z]+", piece.lower())
        hit = any(w in q_tokens for w in words)
        out.append(f"<span class='hit'>{piece}</span>" if hit else piece)
    return "".join(out)


with overview_tab:
    st.subheader("What is this?")
    st.markdown(
        """
        **RAGCopilot** is a Retrieval-Augmented Generation demo. It answers questions
        by first **retrieving** relevant passages from your documents, then **generating**
        a grounded answer from those passages.

        Works out of the box with no API keys. Uses **NVIDIA NIM** for hosted
        frontier open-source LLMs (Kimi K2.6, DeepSeek V4, GLM-5.1, Mistral Large 3,
        Nemotron 3, and more) via a free tier at build.nvidia.com.
        """
    )

    st.subheader("How to use it")
    st.markdown(
        """
        <div class="step"><b>1. Pick a corpus</b> &mdash; toggle the built-in knowledge base
            on (default) or upload your own .txt / .md / .pdf files.</div>
        <div class="step"><b>2. Choose retrieval settings</b> &mdash; chunk size, overlap,
            top-k, and backend. Default TF-IDF works without any downloads.</div>
        <div class="step"><b>3. Choose a generator</b> &mdash; Extractive for
            instant zero-deps answers, or NVIDIA NIM for a hosted frontier LLM
            (Kimi K2.6, DeepSeek V4, GLM-5.1, etc.).</div>
        <div class="step"><b>4. Ask</b> &mdash; type a question in the Ask tab. The app
            shows the top retrieved passages and the generated answer with timings.</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Which option should I pick?")
    guide_df = pd.DataFrame(
        [
            ["If you want...", "Pick", "Works on Streamlit Cloud free tier?"],
            ["Zero deps, instant answers, always works", "Extractive", "Yes"],
            ["Frontier LLM, free hosted", "NVIDIA NIM (Kimi K2.6, DeepSeek V4, etc.)", "Yes - no local deps"],
            ["Lexical search, fastest, no downloads", "TF-IDF backend", "Yes"],
            ["Verify answers aren't hallucinated", "Faithfulness tab", "Yes"],
            ["See retrieval hit/miss on benchmark queries", "Evaluate tab", "Yes"],
        ]
    )
    st.dataframe(guide_df, width="stretch", hide_index=True)

    st.subheader("Quick metrics")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Chunks indexed", f"{len(documents):,}")
    e2.metric("Sources", len(set(source_labels)))
    e3.metric("Backend", "TF-IDF")
    e4.metric("Generator", generator_choice)


def render_answer_card(answer: str) -> None:
    st.markdown(f"<div class='answer'>{answer}</div>", unsafe_allow_html=True)


def render_retrieved(ranked: list[tuple[int, float, str]], query: str) -> None:
    for rank, (idx, score, passage) in enumerate(ranked, start=1):
        with st.expander(
            f"#{rank} &nbsp;&middot;&nbsp; score {score:.3f} &nbsp;&middot;&nbsp; {source_labels[idx]}",
            expanded=(rank == 1),
        ):
            st.markdown(_format_highlighted(passage, query), unsafe_allow_html=True)


def answer_one(question: str, k: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    ranked = retrieve(retriever, question, k)
    retrieval_ms = (time.perf_counter() - t0) * 1000.0
    result = run_generator(
        generator_choice,
        question,
        [r[2] for r in ranked],
        k=k,
        nim_model=NVIDIA_NIM_MODELS.get(nim_model_label) if nim_model_label else None,
    )
    result["retrieval_ms"] = retrieval_ms
    result["ranked"] = ranked
    return result


with ask_tab:
    question = st.text_input("Question", value=SAMPLE_QUESTIONS[0], key="ask_question")
    sample_cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, q in enumerate(SAMPLE_QUESTIONS):
        sample_cols[i].button(
            q[:30] + ("..." if len(q) > 30 else ""),
            use_container_width=True,
            key=f"sample_{i}",
            on_click=_fill_question,
            args=(q,),
        )
    compare = st.toggle(
        "Compare Extractive vs NVIDIA NIM on this question",
        value=False,
        help="Run the same retrieval through both generators and show side-by-side.",
    )

    if not question:
        st.info("Type a question to retrieve context.")
    elif compare:
        ranked = retrieve(retriever, question, top_k)
        passages = [r[2] for r in ranked]
        st.markdown(f"#### Retrieved for: *{question}*")
        render_retrieved(ranked, question)
        st.divider()
        st.markdown("#### Answers")
        cols = st.columns(2)
        nim_id = NVIDIA_NIM_MODELS.get(nim_model_label) or list(NVIDIA_NIM_MODELS.values())[0]
        configs = [
            ("Extractive", ExtractiveGenerator()),
            ("NVIDIA NIM", NvidiaNIMGenerator(model=nim_id)),
        ]
        for col, (label, gen) in zip(cols, configs):
            with col:
                st.markdown(f"##### {label}")
                try:
                    result = gen.generate(question, passages, k=top_k)
                    render_answer_card(result["answer"])
                    st.caption(
                        f"{result['latency_ms']:.1f} ms &middot; "
                        f"{result['generator_name']}"
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{type(exc).__name__}: {exc}")
    else:
        result = answer_one(question, top_k)
        st.markdown("#### Answer")
        if result.get("warning"):
            st.warning(f"Generator unavailable, showing extractive fallback: {result['warning']}")
        render_answer_card(result["answer"])
        st.caption(
            f"retrieval {result['retrieval_ms']:.1f} ms &middot; "
            f"generation {result['latency_ms']:.1f} ms &middot; "
            f"{result['generator_name']} &middot; "
            f"{result['passages_used']} passages"
        )
        st.markdown("#### Retrieved context")
        render_retrieved(result["ranked"], question)


with eval_tab:
    st.subheader("Retrieval evaluation")
    st.markdown(
        "The built-in knowledge base ships with **20 labelled queries**. "
        "Each query's gold passage is known, so we can compute Recall@1, Recall@3, Recall@5, and MRR."
    )
    if not include_demo or not eval_queries:
        st.info("Enable the built-in knowledge base in the sidebar to see metrics.")
    else:
        rows: list[dict[str, Any]] = []
        for query, gold_idx in eval_queries:
            ranked = retrieve(retriever, query, 5)
            top_idxs = [r[0] for r in ranked[:3]]
            hit = gold_idx in top_idxs
            rows.append(
                {
                    "query": query,
                    "gold_passage": source_labels[gold_idx],
                    "top_1": source_labels[top_idxs[0]],
                    "top_2": source_labels[top_idxs[1]] if len(top_idxs) > 1 else "",
                    "top_3": source_labels[top_idxs[2]] if len(top_idxs) > 2 else "",
                    "hit@3": "OK" if hit else "✗",
                }
            )
        df = pd.DataFrame(rows)
        e = corpus_metrics()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall@1", f"{e['recall_at_1']:.1%}")
        c2.metric("Recall@3", f"{e['recall_at_3']:.1%}")
        c3.metric("Recall@5", f"{e['recall_at_5']:.1%}")
        c4.metric("MRR", f"{e['mrr']:.3f}")
        st.dataframe(df, width="stretch", hide_index=True)
        missed = df[df["hit@3"] == "✗"]
        if not missed.empty:
            st.markdown("##### Missed queries")
            st.dataframe(missed, width="stretch", hide_index=True)


with corpus_tab:
    st.subheader("Indexed chunks")
    df = pd.DataFrame(
        {
            "source": source_labels,
            "words": [len(d.split()) for d in documents],
            "preview": [d[:240] + ("..." if len(d) > 240 else "") for d in documents],
        }
    )
    st.dataframe(df, width="stretch", hide_index=True)
    a, b = st.columns(2)
    a.metric("Total chunks", f"{len(documents):,}")
    b.metric("Total words", f"{sum(len(d.split()) for d in documents):,}")


with faith_tab:
    st.subheader("Faithfulness checker")
    st.markdown(
        "Pick a question and a candidate answer. The app shows the token-F1 overlap "
        "between the answer and the best-matching retrieved passage."
    )
    fc1, fc2 = st.columns([2, 1])
    question = fc1.text_input("Question", value=SAMPLE_QUESTIONS[0], key="faith_q")
    k_check = fc2.slider("Top-k", 1, 5, 3, key="faith_k")
    candidate = st.text_area(
        "Candidate answer",
        value="RAG grounds LLMs in retrieved documents to reduce hallucination.",
        height=100,
    )
    if question and candidate:
        ranked = retrieve(retriever, question, k_check)
        passages = [r[2] for r in ranked]
        score = faithfulness_score(candidate, passages)
        st.metric("Faithfulness (max token-F1)", f"{score:.2f}")
        if score >= 0.5:
            st.success("Well-grounded in retrieved context.")
        elif score >= 0.2:
            st.warning("Partial overlap &mdash; some claims may not be supported.")
        else:
            st.error("Low overlap &mdash; likely hallucinated.")
        chart_df = pd.DataFrame(
            {
                "passage": [f"#{i+1}" for i in range(len(passages))],
                "token_f1": [token_overlap_f1(candidate, p) for p in passages],
            }
        )
        st.bar_chart(chart_df.set_index("passage"))


st.divider()
with st.expander("Deployment & production notes"):
    st.markdown(
        """
        **Streamlit Community Cloud** (recommended)
        - 1 GB RAM, CPU. Safe stack: **TF-IDF retriever + Extractive generator + NVIDIA NIM**.
        - Push the project folder to GitHub, point Streamlit at `app.py`.
        - No API keys needed for the basic path. Set `NVIDIA_API_KEY` for the NIM generator.

        **Hugging Face Spaces** (alternative)
        - 16 GB RAM, optional free T4 GPU. Use for the dense retriever if needed.

        **Generator options**
        - *Extractive*: returns the top retrieved passage verbatim. Zero deps.
        - *NVIDIA NIM*: hosted frontier open-source LLMs (Kimi K2.6, DeepSeek V4 Pro,
          GLM-5.1, Mistral Large 3, Nemotron 3, Llama 4 Maverick, etc.) via
          the OpenAI-compatible endpoint at `integrate.api.nvidia.com`. Free tier
          at https://build.nvidia.com. Set `NVIDIA_API_KEY`.

        **Production upgrades**
        - Swap TF-IDF for sentence-transformers (`all-MiniLM-L6-v2`).
        - Add a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
        - Plug in a vector database (FAISS, Pinecone, Chroma).
        - Track freshness, citation accuracy, and hallucination rate.
        """
    )
    st.code(
        "pip install -r requirements.txt\n"
        "streamlit run app.py",
        language="bash",
    )