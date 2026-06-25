"""RAG pipeline: corpus, retrieval, answer generation, evaluation, persistence."""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_NIM_MODEL = "moonshotai/kimi-k2.6"
MAX_NEW_TOKENS = 128
TOKEN_RE = re.compile(r"[a-z]+")

NVIDIA_NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Frontier open-source models hosted on NVIDIA's free inference endpoint
# at https://build.nvidia.com. Pick one and set NVIDIA_API_KEY.
NVIDIA_NIM_MODELS: dict[str, str] = {
    "Kimi K2.6 (1T MoE)":              "moonshotai/kimi-k2.6",
    "DeepSeek V4 Pro (1M ctx MoE)":   "deepseek-ai/deepseek-v4-pro",
    "DeepSeek V4 Flash (284B MoE)":   "deepseek-ai/deepseek-v4-flash",
    "GLM-5.1 (flagship MoE)":         "z-ai/glm-5.1",
    "MiniMax M2.7 (230B)":            "minimax/minimax-m2.7",
    "MiniMax M3 Preview (multimodal)": "minimax/minimax-m3",
    "Mistral Large 3 (675B MoE)":     "mistralai/mistral-large-3-675b-instruct-2512",
    "Nemotron 3 Super (120B MoE)":    "nvidia/nemotron-3-super-120b-a12b",
    "Nemotron 3 Ultra (550B MoE)":    "nvidia/nemotron-3-ultra-550b-a55b",
    "GPT-OSS 120B (MoE)":             "openai/gpt-oss-120b",
    "Llama 4 Maverick (17Bx128 MoE)": "meta/llama-4-maverick-17b-128e-instruct",
}


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

DOCS: list[str] = [
    "Supervised learning uses labeled training data where each example has an input and a known output label. The model learns to map inputs to outputs by minimizing a loss function on the training set.",
    "Unsupervised learning finds hidden patterns or structure in unlabeled data. Common tasks include clustering, dimensionality reduction, and density estimation.",
    "Reinforcement learning trains an agent to make decisions by interacting with an environment. The agent receives rewards or penalties for actions and learns a policy that maximizes cumulative reward.",
    "Overfitting occurs when a model learns noise in the training data instead of the underlying signal. Symptoms include high training accuracy but poor test accuracy. Regularisation and early stopping help prevent overfitting.",
    "Cross-validation splits data into k folds, trains on k-1 folds, and evaluates on the held-out fold. It gives a more reliable estimate of model performance than a single train-test split.",
    "Gradient descent iteratively updates model parameters in the direction of the negative gradient of the loss function. The learning rate controls step size; too large can diverge, too small converges slowly.",
    "A confusion matrix tabulates true positives, true negatives, false positives, and false negatives. From these you can derive accuracy, precision, recall, specificity, and F1-score.",
    "The bias-variance tradeoff captures the tension between underfitting (high bias) and overfitting (high variance). Simple models have high bias; complex models have high variance.",
    "Feature engineering transforms raw data into informative predictors. Techniques include scaling, one-hot encoding, binning, interaction terms, and domain-specific aggregates.",
    "Ensemble methods combine multiple weak learners to produce a strong learner. Bagging reduces variance; boosting reduces bias sequentially.",
    "A neural network consists of layers of neurons, each computing a weighted sum of its inputs followed by a non-linear activation function like ReLU or sigmoid.",
    "Convolutional neural networks use convolution kernels to extract spatial features from grid-like data such as images. Pooling layers reduce spatial dimensions progressively.",
    "Recurrent neural networks maintain a hidden state that captures information from previous time steps, making them suitable for sequential data like text and time series.",
    "Transformers replace recurrence with self-attention, allowing parallel processing of all positions in a sequence. They are the backbone of modern LLMs like GPT and BERT.",
    "Transfer learning takes a model pre-trained on a large dataset and fine-tunes it on a smaller domain-specific dataset. It drastically reduces data and compute requirements.",
    "Batch normalisation normalises layer inputs across a mini-batch to stabilise training, allowing higher learning rates and reducing sensitivity to initialisation.",
    "Dropout randomly sets a fraction of neurons to zero during training, forcing the network to learn redundant representations and reducing overfitting.",
    "An autoencoder learns to compress input data into a latent representation and then reconstruct it. Variants include denoising autoencoders and variational autoencoders.",
    "Generative adversarial networks pit a generator against a discriminator. The generator produces synthetic data; the discriminator tries to tell real from fake.",
    "Attention mechanisms let a model focus on relevant parts of the input when producing each output element. Self-attention computes attention over the input itself.",
    "Retrieval-Augmented Generation (RAG) grounds a large language model in external knowledge by retrieving relevant documents before generating an answer. This reduces hallucination and enables access to private or recent data.",
    "A RAG pipeline has three stages: indexing, retrieval, and generation. Indexing chunks and embeds documents. Retrieval finds top-k relevant chunks. Generation feeds context plus query to an LLM.",
    "Chunking splits long documents into smaller passages so retrieval returns focused context. Common strategies include fixed-length token splits, sentence boundaries, and semantic segmentation.",
    "TF-IDF is a sparse retrieval method that weights terms by how often they appear in a document relative to their rarity across the corpus.",
    "Dense retrieval uses embedding models to encode queries and documents into dense vectors. Cosine similarity between embeddings finds semantically relevant passages.",
    "A hybrid retriever combines sparse and dense methods, often with a weighted sum or reciprocal rank fusion, to get the best of both lexical and semantic matching.",
    "Reranking uses a cross-encoder model that jointly processes query and candidate passage to produce a more accurate relevance score. Applied to top results from a first-stage retriever.",
    "Retrieval evaluation metrics include Recall at k, Mean Reciprocal Rank, NDCG, and precision at k. Each captures a different aspect of retrieval quality.",
    "Grounding means the generated answer must be verifiable against the retrieved context. Techniques include citation extraction and faithfulness classification.",
    "Hallucination in LLMs occurs when the model generates plausible-sounding but factually incorrect content. RAG mitigates this by constraining generation to retrieved evidence.",
    "A vector database stores embeddings and supports efficient approximate nearest neighbour search. Popular options include FAISS, Pinecone, Weaviate, and Chroma.",
    "Prompt injection is a security attack where a user crafts input to override the system prompt. Guardrails like input filtering and output validation reduce risk.",
    "An agentic RAG system extends basic RAG by letting the LLM call tools, issue multiple queries, synthesise results, and decide when external retrieval is needed.",
    "Value at Risk estimates the maximum loss not exceeded with a given confidence level over a specified horizon. A 95 percent daily VaR of one million means a 5 percent chance of losing more than that.",
    "Expected Shortfall averages the losses that occur beyond the VaR threshold. It gives a more complete picture of tail risk than VaR alone.",
    "Credit scoring models estimate the probability that a borrower will default. Algorithms include logistic regression, gradient boosting, and neural networks.",
    "Fraud detection systems flag transactions that deviate from normal patterns. They are evaluated on precision and recall with a cost-sensitive threshold.",
    "Anti-money laundering systems screen transactions against watchlists and detect suspicious patterns like structuring and rapid movement of funds.",
    "Portfolio optimisation selects asset weights to maximise expected return for a given risk level using mean-variance optimisation and the efficient frontier.",
    "The Sharpe ratio measures risk-adjusted return as portfolio return minus risk-free rate divided by portfolio volatility.",
    "Monte Carlo simulation draws random samples from probability distributions to model uncertainty in financial forecasts and option pricing.",
    "Time series forecasting uses models like ARIMA and Prophet. Features include lagged values, rolling statistics, and calendar effects.",
    "Model risk management monitors ML models for data drift, concept drift, performance degradation, and fairness violations.",
    "Medical image segmentation partitions an image into regions of interest such as tumours or organs. U-Net is the standard architecture for biomedical segmentation.",
    "Digital pathology scans whole-slide images of tissue at high resolution. AI models assist pathologists by detecting and grading abnormalities in H&E-stained slides.",
    "Gleason grading is a histological scoring system for prostate cancer ranging from 3 to 5. AI models automate Gleason pattern recognition from biopsy slides.",
    "ISUP grade groups condense Gleason scores into five prognostic categories. Grade Group 1 is Gleason 3+3 while Grade Group 5 is Gleason 9-10.",
    "Dice score and Intersection-over-Union are standard metrics for segmentation. Dice measures pixel overlap between predicted and ground-truth masks.",
    "A heatmap overlay on a pathology slide shows where the model detects abnormality. High-intensity regions flag suspicious areas for pathologist review.",
    "Stain normalisation standardises colour variation in histology images caused by different labs and scanners. It improves model generalisation across institutions.",
    "Whole-slide image tiling splits a gigapixel slide into smaller patches for processing. Tiling strategies use tissue-detection masks to skip background regions.",
    "Predictive toxicology uses AI to forecast compound toxicity from chemical structure and multi-modal data, reducing reliance on animal testing.",
    "Medical AI safety disclaimers are essential: model outputs are research demonstrations and require clinician review before any clinical decision.",
    "An ML pipeline orchestrates data ingestion, validation, preprocessing, training, evaluation, and deployment. Tools like Kubeflow, Airflow, and ZenML manage orchestration.",
    "Model monitoring tracks prediction distributions, feature drift, and performance metrics in production. Alerts trigger investigation or retraining when drift exceeds thresholds.",
    "A/B testing for ML compares a new model against the current production model on live traffic. Metrics like conversion rate and latency determine which wins.",
    "CI/CD for ML automates testing and deployment of models. Unit tests validate data transforms; integration tests run the full pipeline.",
    "Containerisation with Docker packages model code and dependencies into a portable image. Kubernetes orchestrates container deployment at scale.",
    "Feature stores centralise feature definitions and computation so training and serving use identical features. Tecton and Feast are popular options.",
    "SHAP explains individual predictions by computing each feature contribution. It is used for model debugging, regulatory compliance, and stakeholder trust.",
    "Synthetic data generation creates artificial datasets that preserve the statistical properties of real data. Useful for testing and privacy protection.",
    "Model compression techniques like pruning, quantisation, and knowledge distillation shrink models for edge deployment with minimal accuracy loss.",
    "Streamlit Community Cloud hosts Streamlit apps for free with 1GB RAM. Hugging Face Spaces offers more generous limits and free GPU tiers.",
]

SOURCES: list[str] = [
    "ml_basics.md",
    "ml_basics.md",
    "ml_basics.md",
    "ml_basics.md",
    "ml_basics.md",
    "ml_basics.md",
    "ml_basics.md",
    "ml_basics.md",
    "ml_basics.md",
    "ml_basics.md",
    "deep_learning.md",
    "deep_learning.md",
    "deep_learning.md",
    "deep_learning.md",
    "deep_learning.md",
    "deep_learning.md",
    "deep_learning.md",
    "deep_learning.md",
    "deep_learning.md",
    "deep_learning.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "rag_genai.md",
    "finance_risk.md",
    "finance_risk.md",
    "finance_risk.md",
    "finance_risk.md",
    "finance_risk.md",
    "finance_risk.md",
    "finance_risk.md",
    "finance_risk.md",
    "finance_risk.md",
    "finance_risk.md",
    "medical_ai.md",
    "medical_ai.md",
    "medical_ai.md",
    "medical_ai.md",
    "medical_ai.md",
    "medical_ai.md",
    "medical_ai.md",
    "medical_ai.md",
    "medical_ai.md",
    "medical_ai.md",
    "mlops.md",
    "mlops.md",
    "mlops.md",
    "mlops.md",
    "mlops.md",
    "mlops.md",
    "mlops.md",
    "mlops.md",
    "mlops.md",
    "mlops.md",
]

EVAL_QUERIES: list[tuple[str, int]] = [
    ("What is supervised learning?", 0),
    ("How does cross-validation work?", 4),
    ("What is the bias-variance tradeoff?", 7),
    ("How do transformers work?", 13),
    ("What is RAG and how does it help?", 20),
    ("What are the stages of a RAG pipeline?", 21),
    ("How does TF-IDF work?", 23),
    ("What is reranking?", 26),
    ("How is retrieval evaluated?", 27),
    ("What is hallucination in LLMs?", 29),
    ("What is Value at Risk?", 33),
    ("What is Expected Shortfall?", 34),
    ("How does credit scoring work?", 35),
    ("How is fraud detection evaluated?", 36),
    ("What is the Sharpe ratio?", 39),
    ("What is U-Net used for in medical imaging?", 40),
    ("How does Gleason grading work?", 42),
    ("What are Dice and IoU metrics?", 44),
    ("What is model monitoring?", 51),
    ("What is SHAP used for?", 56),
]


def demo_corpus() -> dict[str, Any]:
    return {"docs": DOCS, "sources": SOURCES, "queries": EVAL_QUERIES}


def chunk_text(text: str, chunk_words: int = 120, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def read_pdf(file) -> str:
    """Extract text from a PDF file-like object. Returns empty string on failure."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            file.seek(0)
            return file.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""
    try:
        reader = PdfReader(file)
        parts: list[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts)
    except Exception:
        return ""


def read_uploads(
    uploaded_files,
    chunk_words: int = 120,
    overlap_words: int = 30,
) -> tuple[list[str], list[str]]:
    docs: list[str] = []
    sources: list[str] = []
    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        suffix = Path(name).suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = read_pdf(uploaded_file)
        else:
            continue
        for chunk_id, chunk in enumerate(chunk_text(text, chunk_words, overlap_words), start=1):
            docs.append(chunk)
            sources.append(f"{name} · chunk {chunk_id}")
    return docs, sources


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_tfidf(docs: list[str]) -> dict[str, Any]:
    """Sparse TF-IDF retriever. Stores a unit-normalised document matrix."""
    token_lists = [_tokenize(d) for d in docs]
    df: dict[str, int] = {}
    for tokens in token_lists:
        for word in set(tokens):
            df[word] = df.get(word, 0) + 1
    n = len(docs)
    idf = {w: math.log((1 + n) / (1 + dfv)) + 1.0 for w, dfv in df.items()}
    vocab_keys = list(idf.keys())
    ix = {w: i for i, w in enumerate(vocab_keys)}
    matrix = np.zeros((n, len(vocab_keys)), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        if not tokens:
            continue
        counts: dict[str, int] = {}
        for w in tokens:
            counts[w] = counts.get(w, 0) + 1
        for w, c in counts.items():
            if w in ix:
                matrix[i, ix[w]] = c * idf[w]
        norm = np.linalg.norm(matrix[i])
        if norm > 0:
            matrix[i] /= norm
    return {"M": matrix, "idf": idf, "ix": ix, "docs": docs, "kind": "tfidf"}


def score(model: dict[str, Any], query: str) -> np.ndarray:
    vec = np.zeros(len(model["idf"]), dtype=np.float32)
    for w in _tokenize(query):
        if w in model["ix"]:
            vec[model["ix"][w]] += model["idf"][w]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return model["M"] @ vec


def retrieve(
    model: dict[str, Any],
    query: str,
    k: int = 3,
) -> list[tuple[int, float, str]]:
    """Return the top-k (index, score, passage) tuples for a query."""
    scores = score(model, query)
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i]), model["docs"][i]) for i in order]


def evaluate_retrieval(
    model: dict[str, Any],
    queries: list[tuple[str, int]],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"n_queries": len(queries)}
    if not queries:
        for k in ks:
            metrics[f"recall_at_{k}"] = 0.0
        metrics["mrr"] = 0.0
        return metrics
    max_k = max(ks)
    hit_counts = {k: 0 for k in ks}
    rr_sum = 0.0
    latencies_ms: list[float] = []
    for query, gold_index in queries:
        t0 = time.perf_counter()
        ranked = retrieve(model, query, max_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        for k in ks:
            if gold_index in [r[0] for r in ranked[:k]]:
                hit_counts[k] += 1
        for rank, (idx, _, _) in enumerate(ranked, start=1):
            if idx == gold_index:
                rr_sum += 1.0 / rank
                break
    for k in ks:
        metrics[f"recall_at_{k}"] = hit_counts[k] / len(queries)
    metrics["mrr"] = rr_sum / len(queries)
    metrics["avg_latency_ms"] = float(np.mean(latencies_ms))
    return metrics


def build_retriever(docs: list[str]) -> dict[str, Any]:
    return build_tfidf(docs)


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def token_overlap_f1(answer: str, context: str) -> float:
    a = set(TOKEN_RE.findall(answer.lower()))
    c = set(TOKEN_RE.findall(context.lower()))
    if not a or not c:
        return 0.0
    common = a & c
    if not common:
        return 0.0
    precision = len(common) / len(a)
    recall = len(common) / len(c)
    return 2 * precision * recall / (precision + recall)


def faithfulness_score(answer: str, retrieved_passages: list[str]) -> float:
    if not answer or not retrieved_passages:
        return 0.0
    return float(max(token_overlap_f1(answer, p) for p in retrieved_passages))


def highlight_tokens(text: str, query: str) -> list[tuple[str, bool]]:
    q_tokens = set(TOKEN_RE.findall(query.lower()))
    pieces: list[tuple[str, bool]] = []
    for piece in re.split(r"(\s+)", text):
        if not piece:
            continue
        if piece.isspace():
            pieces.append((piece, False))
            continue
        hit = any(w in q_tokens for w in TOKEN_RE.findall(piece.lower()))
        pieces.append((piece, hit))
    return pieces


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _format_prompt(question: str, passages: list[str], k: int) -> str:
    context = "\n".join(f"[{i}] {p.strip()}" for i, p in enumerate(passages[:k], start=1))
    return (
        "You are a grounded assistant. Answer the question using ONLY the context below. "
        "If the answer is not in the context, say 'I do not know based on the provided context.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer:"
    )


class ExtractiveGenerator:
    """Returns the top retrieved passage verbatim. Zero dependencies, zero latency."""

    name = "extractive"

    def generate(self, question: str, passages: list[str], k: int = 1) -> dict[str, Any]:
        if not passages:
            return {"answer": "No context available.", "passages_used": 0, "latency_ms": 0.0}
        top = passages[:k]
        return {
            "answer": " ".join(top),
            "passages_used": len(top),
            "latency_ms": 0.0,
            "generator_name": self.name,
        }


class NvidiaNIMGenerator:
    def __init__(self, model: str = "moonshotai/kimi-k2.6") -> None:
        self.model = model
        self._api_key = None

    def _ensure_loaded(self) -> None:
        if self._api_key is not None:
            return
        self._api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("NVIDIA_API_KEY env var is not set. Get a free key at https://build.nvidia.com.")

    @property
    def name(self) -> str:
        return f"nim:{self.model.split('/')[-1]}"

    def generate(self, question: str, passages: list[str], k: int = 3) -> dict[str, Any]:
        import requests as req

        self._ensure_loaded()
        context = "\n\n".join(f"[{i}] {p.strip()}" for i, p in enumerate(passages[:k], start=1))
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question.strip()}\n\n"
                        "Answer using ONLY the provided context. "
                        "If the answer is not in the context, say "
                        "'I do not know based on the provided context.' Cite passages using [n]."
                    ),
                }
            ],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }
        t0 = time.perf_counter()
        resp = req.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = (data["choices"][0]["message"]["content"] or "").strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not answer:
            answer = "I do not know based on the provided context."
        return {
            "answer": answer,
            "passages_used": min(k, len(passages)),
            "latency_ms": latency_ms,
            "generator_name": self.name,
        }


def make_generator(kind: str) -> Any:
    kind = kind.lower()
    if kind.startswith("nim") or kind.startswith("nvidia"):
        return NvidiaNIMGenerator()
    return ExtractiveGenerator()


def run_generator(
    choice: str,
    question: str,
    passages: list[str],
    k: int,
    nim_model: str | None = None,
) -> dict[str, Any]:
    """Run the chosen generator, falling back to extractive on any failure."""
    try:
        if choice.startswith("NVIDIA NIM") or choice.lower().startswith("nim"):
            gen = NvidiaNIMGenerator(model=nim_model) if nim_model else NvidiaNIMGenerator()
        else:
            gen = ExtractiveGenerator()
        return gen.generate(question, passages, k=k)
    except Exception as exc:  # noqa: BLE001
        result = ExtractiveGenerator().generate(question, passages, k=k)
        result["warning"] = f"{type(exc).__name__}: {exc}"
        result["generator_name"] = "extractive-fallback"
        return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_retriever(model: dict[str, Any], path: str | Path = "models/model.pkl") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    safe = {k: v for k, v in model.items() if k != "encoder"}
    with open(out, "wb") as f:
        pickle.dump(safe, f)
    return out


def load_retriever(path: str | Path = "models/model.pkl") -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(metrics: dict[str, Any], path: str | Path = "models/metrics.json") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))


def load_metrics(path: str | Path = "models/metrics.json") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def print_report(metrics: dict[str, Any]) -> None:
    print("=" * 50)
    print("RAGCopilot Retrieval Evaluation")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<22}: {value:.4f}")
        else:
            print(f"  {key:<22}: {value}")