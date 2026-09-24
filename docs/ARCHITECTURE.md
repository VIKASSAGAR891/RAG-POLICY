# Architecture Notes

## Request path

1. PDFs are parsed page-by-page with PyMuPDF.
2. A custom section-aware chunker identifies common legal headings and preserves section metadata.
3. BGE embeddings are generated locally with SentenceTransformers.
4. FAISS performs dense retrieval.
5. BM25 performs lexical retrieval.
6. Reciprocal Rank Fusion combines both candidate lists.
7. A BGE cross-encoder reranks the candidate pool.
8. LlamaIndex's Groq integration calls the hosted LLM.
9. The prompt explicitly restricts generation to retrieved evidence.
10. Source metadata is returned with the answer.
11. RAGAS evaluates generated responses and retrieved contexts.

## Why LlamaIndex is present

LlamaIndex is used at the LLM integration layer through its Groq adapter and is intentionally kept visible in the architecture rather than hiding the retrieval implementation behind a single high-level query engine. This keeps the hybrid retrieval and reranking logic explicit for evaluation.

## Experimental baseline

The current pipeline is the full system. To run controlled experiments, set retrieval/reranking switches in a future experiment runner or temporarily call the underlying `HybridIndex` methods directly. The repository keeps the retrieval layers separate so the experiment runner can compare dense-only, hybrid, and reranked configurations without changing document ingestion.
