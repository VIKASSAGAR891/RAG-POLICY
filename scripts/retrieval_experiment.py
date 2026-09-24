"""
Retrieval-only experiment runner.

It compares:
1. Dense retrieval
2. BM25
3. Hybrid retrieval (RRF)
4. Hybrid + BGE reranking

The output is a JSON file containing ranked source IDs. This lets you
measure retrieval changes independently before paying for LLM calls.
"""

import json
import sys
from pathlib import Path

from src.policy_qa.settings import SETTINGS
from src.policy_qa.embeddings import BGEEmbedder
from src.policy_qa.index import HybridIndex
from src.policy_qa.reranker import BGEReranker


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python scripts/retrieval_experiment.py "your question"')

    embedder = BGEEmbedder(SETTINGS.embedding_model)
    index = HybridIndex.load(SETTINGS.index_dir, embedder, SETTINGS.rrf_k)

    dense = index.dense_search(question, SETTINGS.dense_top_k)
    sparse = index.bm25_search(question, SETTINGS.bm25_top_k)
    hybrid = index.hybrid_search(question, SETTINGS.dense_top_k, SETTINGS.bm25_top_k)

    reranker = BGEReranker(SETTINGS.reranker_model)
    reranked = reranker.rerank(question, hybrid, SETTINGS.rerank_top_k)

    def summarize(items):
        out = []
        for item in items:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
                idx, score = item
                c = index.chunks[idx]
                out.append({"chunk_id": c.chunk_id, "document": c.document,
                            "section": c.section_id, "page": c.page_start,
                            "score": float(score)})
            else:
                c = item["chunk"]
                out.append({"chunk_id": c.chunk_id, "document": c.document,
                            "section": c.section_id, "page": c.page_start,
                            "retrieval_score": item["retrieval_score"],
                            "rerank_score": item["rerank_score"]})
        return out

    result = {
        "question": question,
        "dense": summarize(dense),
        "bm25": summarize(sparse),
        "hybrid": summarize(hybrid[:SETTINGS.rerank_top_k]),
        "hybrid_reranked": summarize(reranked),
    }

    out = Path("evals") / "retrieval_experiment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
