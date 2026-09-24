from sentence_transformers import CrossEncoder

class BGEReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        pairs = [(query, c.text) for c, _ in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        output = []
        for (chunk, retrieval_score), score in ranked[:top_k]:
            output.append({
                "chunk": chunk,
                "retrieval_score": float(retrieval_score),
                "rerank_score": float(score),
            })
        return output
