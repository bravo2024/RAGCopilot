# RAGCopilot

A Streamlit RAG demo: upload documents, ask questions, get grounded answers from NVIDIA NIM frontier LLMs. Works on Streamlit Cloud free tier, no GPU needed.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Set `NVIDIA_API_KEY` in Streamlit secrets to use the NIM models (get a free key at https://build.nvidia.com).

## What's in the box

- **Retriever**: TF-IDF (zero extra deps, builds on startup)
- **Two generators**: extractive (returns top retrieved passage verbatim) and NVIDIA NIM (Kimi K2.6, DeepSeek V4 Pro, GLM-5.1, Mistral Large 3, Nemotron 3, Llama 4 Maverick, etc.)
- **20 labeled eval queries** with Recall@1, Recall@3, Recall@5, MRR
- **Faithfulness checker** that measures token-F1 overlap between answer and retrieved context
- **63-doc demo corpus** covering ML basics, deep learning, RAG/GenAI, finance, medical AI, and MLOps
- **Upload your own** .txt, .md, or .pdf files

## Tabs

- **Ask** — type a question, pick a generator, see the answer with highlighted citations and latency
- **Evaluate** — run all 20 eval queries, see headline metrics and per-query hit/miss
- **Corpus** — browse every indexed chunk, source label, and word count
- **Faithfulness** — paste any answer, see its token-F1 score against retrieved context, get a grounded/partial/hallucinated label

## Project structure

```
RAGCopilot/
  app.py                  Streamlit app
  src/
    rag.py                corpus, retrieval, NIM generation, faithfulness, persistence
    __init__.py
  data/raw/               63 source markdown files
  models/
    metrics.json          saved eval metrics
  requirements.txt
  runtime.txt             python-3.11
  .streamlit/config.toml
```

## Evaluation (TF-IDF, bundled 63 docs)

| Metric | Value |
|--------|------:|
| Recall@1 | 0.60 |
| Recall@3 | 0.70 |
| Recall@5 | 0.70 |
| MRR | 0.64 |


## Deploy

1. Push to GitHub
2. Go to https://share.streamlit.io, click New app, pick your repo, main file = `app.py`
3. Add `NVIDIA_API_KEY` in Streamlit secrets if you want NIM generation


