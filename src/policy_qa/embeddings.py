from sentence_transformers import SentenceTransformer
import numpy as np

class BGEEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts):
        # BGE recommends a query instruction for queries; document embeddings
        # can be generated directly.
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_query(self, query):
        instruction = "Represent this sentence for searching relevant passages: "
        vec = self.model.encode(
            instruction + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vec, dtype="float32")
