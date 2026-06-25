# RAG and GenAI

## Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) grounds a large language model in external knowledge by retrieving relevant documents before generating an answer. This reduces hallucination and enables access to private or recent data.

## RAG pipeline stages

A RAG pipeline has three stages: indexing, retrieval, and generation. Indexing chunks and embeds documents. Retrieval finds top-k relevant chunks. Generation feeds context plus query to an LLM.

## Chunking strategies

Chunking splits long documents into smaller passages so retrieval returns focused context. Common strategies include fixed-length token splits, sentence boundaries, and semantic segmentation.

## TF-IDF retrieval

TF-IDF is a sparse retrieval method that weights terms by how often they appear in a document relative to their rarity across the corpus.

## Dense retrieval

Dense retrieval uses embedding models to encode queries and documents into dense vectors. Cosine similarity between embeddings finds semantically relevant passages.

## Hybrid retrieval

A hybrid retriever combines sparse and dense methods, often with a weighted sum or reciprocal rank fusion, to get the best of both lexical and semantic matching.

## Reranking

Reranking uses a cross-encoder model that jointly processes query and candidate passage to produce a more accurate relevance score. Applied to top results from a first-stage retriever.

## Retrieval evaluation

Retrieval evaluation metrics include Recall at k, Mean Reciprocal Rank, NDCG, and precision at k. Each captures a different aspect of retrieval quality.

## Grounding

Grounding means the generated answer must be verifiable against the retrieved context. Techniques include citation extraction and faithfulness classification.

## Hallucination

Hallucination in LLMs occurs when the model generates plausible-sounding but factually incorrect content. RAG mitigates this by constraining generation to retrieved evidence.

## Vector databases

A vector database stores embeddings and supports efficient approximate nearest neighbour search. Popular options include FAISS, Pinecone, Weaviate, and Chroma.

## Prompt injection

Prompt injection is a security attack where a user crafts input to override the system prompt. Guardrails like input filtering and output validation reduce risk.

## Agentic RAG

An agentic RAG system extends basic RAG by letting the LLM call tools, issue multiple queries, synthesise results, and decide when external retrieval is needed.
